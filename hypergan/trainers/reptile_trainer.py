import numpy as np
import torch
import hyperchamber as hc
import inspect
import copy
import math
import os
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from hypergan.trainers.base_trainer import BaseTrainer
from hypergan.optimizers.adamirror import Adamirror

TINY = 1e-12

def ng(param, gan):
    grad = param.grad
    if grad is None:
        if gan.steps == 0:
            print("Warning: missing gradient for " + str(param.shape))
        return None
    grad_norm = grad.norm()
    max_norm = (1e-2) * torch.maximum(param.norm(),torch.ones([], device=param.device)*1e-8)
    trigger = grad_norm < max_norm
    ## This little max(., 1e-6) is distinct from the normal eps and just prevents
    ## division by zero. It technically should be impossible to engage.
    clipped_grad = grad * (max_norm / torch.maximum(grad_norm, torch.ones([], device=param.device)*1e-6))
    return torch.where(trigger, grad, clipped_grad)

class ReptileTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = self.create_optimizer()
        self.ttur = self.config.ttur or 1.0
        self.relu = torch.nn.ReLU()
        self.task_count = 0
        if self.config.post_ode_hooks:
            self.post_ode_hooks = self.gan.setup_hooks(config_name="post_ode_hooks", add_to_hooks=False)
        self.meta_gan = hg.GAN(config=self.gan.config, inputs=self.gan.inputs, device=self.gan.device) #TODO load/save
        self.meta_optimizer = self.create_optimizer(trainable_gan=False, parameters=self.meta_gan.parameters())
        self.reptile_save_file = "/".join(self.trainable_gan.save_file.split("/")[:-1])+"-reptile/save"
        success = self.load_meta_gan()
        print("Loading reptile gan", success)

    def required(self):
        return "optimizer".split()

    def set_parameter_grads(self, optimizer, task, add_train_hooks=True, first_step=False):
        d_grads, g_grads = self.calculate_gradients(optimizer, task, add_train_hooks=add_train_hooks, first_step=first_step)

        for hook in self.train_hooks:
            if not first_step or hook.config.first_ode_step_only != True:
                d_grads, g_grads = hook.gradients(d_grads, g_grads)
        for p, np in zip(self.trainable_gan.d_parameters(), d_grads):
            p.grad = np
            if self.config.adaptive_gradient_norm:
                p.grad = ng(p, self.gan)
        for p, np in zip(self.trainable_gan.g_parameters(), g_grads):
            p.grad = np
            if self.config.adaptive_gradient_norm:
                p.grad = ng(p, self.gan)

        #if self.config.gradient_max_norm:
        #    torch.nn.utils.clip_grad_norm_(self.trainable_gan.parameters(), self.config.gradient_max_norm)

        return d_grads, g_grads

    def inner_step(self, feed_dict, gan, optimizer, task):
        gan._metrics = {}
        metrics = gan.metrics()

        self.before_step(self.current_step, feed_dict)
        gan.next_inputs()

        self.set_parameter_grads(optimizer, task)

        self.optimizer.step()
        self.after_step(self.current_step, feed_dict)

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)

    def _step(self, feed_dict):
        inner_steps=2
        num_tasks = 2
        for c, mc in zip(self.gan.components(), self.meta_gan.components()):
            c.train()
            mc.train()
        # copy weights to active gan
        for c, mc in zip(self.gan.components(), self.meta_gan.components()):
            c.load_state_dict(mc.state_dict())

        task = self.task_count % num_tasks
        # optimize active gan
        for i in range(inner_steps):
            self.inner_step(feed_dict, self.gan, self.optimizer, task)

        self.task_count += 1
        self.meta_optimizer.zero_grad()
        # update gradients on reptile meta gan
        for mc, c in zip(self.meta_gan.components(), self.gan.components()):
            for p, ap in zip(mc.parameters(), c.parameters()):
                diff = p - ap
                p.grad = diff

        self.meta_optimizer.step()

        if self.task_count % 1000 == 0:
            print("Saving reptile")
            self.save_meta_gan()

    def save_meta_gan(self):
        full_path = os.path.expanduser(os.path.dirname(self.reptile_save_file))
        os.makedirs(full_path, exist_ok=True)
        self.meta_gan.save(full_path)
        self.meta_gan._save(full_path, "optimizer_reptile", self.meta_optimizer)

    def load_meta_gan(self):
        success = self.meta_gan.load(self.reptile_save_file)
        full_path = os.path.expanduser(os.path.dirname(self.reptile_save_file))
        self.meta_gan._load(full_path, "optimizer_reptile", self.meta_optimizer)
        return success


    def calculate_gradients(self, optimizer, task, add_train_hooks=True, create_graph=False, first_step=False):
        optimizer.zero_grad()
        d_loss, g_loss = self.trainable_gan.forward_loss(task)
        self.gan.add_metric('d_loss', d_loss.mean())
        self.gan.add_metric('g_loss', g_loss.mean())
        if add_train_hooks:
            for hook in self.train_hooks:
                if not first_step or hook.config.first_ode_step_only != True:
                    loss = hook.forward(d_loss, g_loss)
                    if hook.config.only:
                        if loss[0] is not None:
                            d_loss = loss[0]
                        if loss[1] is not None:
                            g_loss = loss[1]
                    else:
                        if loss[0] is not None:
                            d_loss = d_loss + loss[0]
                        if loss[1] is not None:
                            g_loss = g_loss + loss[1]
        for loss_function in self.gan.additional_losses:
            loss = loss_function()
            if loss[0] is not None:
                d_loss = d_loss + loss[0]
            if loss[1] is not None:
                g_loss = g_loss + loss[1]

        if self.config.joint:
            g_loss.mean().backward(retain_graph=True, create_graph=create_graph)
            d_loss.mean().backward(retain_graph=True)
        else:

            self.trainable_gan.set_generator_trainable(True)
            self.trainable_gan.set_discriminator_trainable(False)

            g_loss.mean().backward(retain_graph=True, create_graph=create_graph)

            self.trainable_gan.set_generator_trainable(False)
            self.trainable_gan.set_discriminator_trainable(True)

            d_loss.mean().backward(retain_graph=True)
            self.trainable_gan.set_generator_trainable(True)

        for loss, optim in self.gan.additional_optimizer_steps:
            optim.zero_grad()
            loss().backward()
            optim.step()


        d_grads = [p.grad for p in self.trainable_gan.d_parameters()]
        g_grads = [p.grad for p in self.trainable_gan.g_parameters()]

        #gnd = sum([p.grad.norm() for p in self.trainable_gan.d_parameters()])
        #self.gan.add_metric("GNd", gnd)
        #gng = sum([p.grad.norm() for p in self.trainable_gan.g_parameters()])
        #self.gan.add_metric("GNg", gng)
        return d_grads, g_grads

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))
        for value in metric_values:
            if math.isnan(value):
                print("NAN detected, exitting")
                os._exit(1)

