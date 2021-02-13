import numpy as np
import torch
import hyperchamber as hc
import inspect
import copy
import math

from hypergan.gan_component import ValidationException, GANComponent
from hypergan.trainers.base_trainer import BaseTrainer
from hypergan.optimizers.adamirror import Adamirror
from hypergan.optimizers.sam import SAM

TINY = 1e-12

class SimultaneousTrainer(BaseTrainer):
    """ Steps G and D simultaneously """
    def _create(self):
        self.optimizer = self.create_optimizer()
        self.ttur = self.config.ttur or 1.0

    def required(self):
        return "optimizer".split()

    def set_parameter_grads(self, add_train_hooks=True, first_step=False):
        d_grads, g_grads = self.calculate_gradients(add_train_hooks=add_train_hooks, first_step=first_step)

        for hook in self.train_hooks:
            if not first_step or hook.config.first_ode_step_only != True:
                d_grads, g_grads = hook.gradients(d_grads, g_grads)
        for p, np in zip(self.trainable_gan.d_parameters(), d_grads):
            p.grad = np
        for p, np in zip(self.trainable_gan.g_parameters(), g_grads):
            p.grad = np

        return d_grads, g_grads

    def _step(self, feed_dict):
        self.gan._metrics = {}
        metrics = self.gan.metrics()

        self.before_step(self.current_step, feed_dict)
        self.gan.next_inputs()

        if self.config.ode_solver == 'rk4':
            self.rk4_ode_step()
            self.optimizer.step()
            self.print_metrics(self.current_step)
            return 
        if self.config.ode_solver == 'heun':
            gn = 100.0
            self.gan._metrics = {}
            self.heun_ode_step()
            gnd = sum([p.grad.norm() for p in self.trainable_gan.d_parameters()])
            self.gan.add_metric("GNdf", gnd)
            gng = sum([p.grad.norm() for p in self.trainable_gan.d_parameters()])
            self.gan.add_metric("GNgf", gng)
            if self.config.gradient_max_norm:
                torch.nn.utils.clip_grad_norm_(self.trainable_gan.parameters(), self.config.gradient_max_norm)
            self.optimizer.step()
            #if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)
            return

        self.set_parameter_grads()

        if(isinstance(self.optimizer, SAM)):
            self.optimizer.first_step()
            d_grads, g_grads = self.calculate_gradients()

            for hook in self.train_hooks:
                d_grads, g_grads = hook.gradients(d_grads, g_grads)
            for p, np in zip(self.trainable_gan.d_parameters(), d_grads):
                p.grad = np * self.ttur
            for p, np in zip(self.trainable_gan.g_parameters(), g_grads):
                p.grad = np

            if self.config.gradient_max_norm:
                torch.nn.utils.clip_grad_norm_(self.trainable_gan.parameters(), self.config.gradient_max_norm)

            self.optimizer.second_step()
        else:
            self.optimizer.step()

        if self.current_step % 10 == 0:
            self.print_metrics(self.current_step)

    # deep copies model + grads of model
    def grad_clone(self, source: torch.nn.Module) -> torch.nn.Module:
        dest = copy.deepcopy(source)
        dest.requires_grad_(True)

        for s_p, d_p in zip(source.parameters(), dest.parameters()):
            if s_p.grad is not None:
                d_p.grad = s_p.grad.clone()

        return dest


    def ode_gan_step(self, G, D, data, detach_err: bool = True, retain_graph: bool = False, add_train_hooks = True, first_step = False):
        oldd = self.gan.discriminator.net
        oldg = self.gan.generator.net
        self.gan.discriminator.net = D
        self.gan.generator.net = G
        errD, errG = self.trainable_gan.forward_loss()
        d_grads, g_grads = self.set_parameter_grads(add_train_hooks=add_train_hooks, first_step=first_step)

        DISC_GRAD_CACHE = self.grad_clone(self.gan.discriminator.net)
        GEN_GRAD_CACHE = self.grad_clone(self.gan.generator.net)

        if detach_err:
            errG = errG.detach()
            errD = errD.detach()

        self.gan.discriminator.net = oldd
        self.gan.generator.net = oldg
        return DISC_GRAD_CACHE, GEN_GRAD_CACHE, errD, errG, None, None, None

    # Heun's ODE Step
    def heun_ode_step(self):
        G = self.gan.generator
        D = self.gan.discriminator
        data = self.gan.x
        step_size = self.config.ode_step_size or 0.1
        disc_reg = 0.01
        # Compute first step of Heun
        theta_1, phi_1, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(G.net, D.net, data, detach_err=False, retain_graph=True, add_train_hooks=True, first_step=True)


        # Compute the L2 norm using the prior computation graph
        grad_norm = None
        for phi_0_param in G.parameters():
            if phi_0_param.grad is not None:
                if grad_norm is None:
                    grad_norm = phi_0_param.grad.square().sum()
                else:
                    grad_norm = grad_norm + phi_0_param.grad.square().sum()

        grad_norm = grad_norm.sqrt()

        # Preserve gradients for regularization in cache
        #D_norm_grads = torch.autograd.grad(grad_norm, list(D.parameters()))
        D_norm_grads = []

        grad_norm = grad_norm.detach()

        # Compute norm of the gradients of the discriminator for logging
        disc_grad_norm = torch.tensor(0.0, device="cuda:0")
        for d_grad, in zip(D_norm_grads):
            # compute discriminator norm
            disc_grad_norm = disc_grad_norm + d_grad.detach().square().sum().sqrt()

        # Detach graph
        errD = errD.detach()
        errG = errG.detach()

        # preserve theta, phi for next computation
        theta_0 = self.grad_clone(theta_1)
        phi_0 = self.grad_clone(phi_1)

        # Update theta and phi for first heun step]
        for d_param, theta_1_param in zip(D.parameters(), theta_1.parameters()):
            if theta_1_param.grad is not None:
                theta_1_param.data = d_param.data + (step_size * -theta_1_param.grad)

        for g_param, phi_1_param in zip(G.parameters(), phi_1.parameters()):
            if phi_1_param.grad is not None:
                phi_1_param.data = g_param.data + (step_size * -phi_1_param.grad)

        # Compute second step of Heun
        theta_2, phi_2, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(phi_1, theta_1, data, add_train_hooks=True)
        def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
            # normalize gradient
            grad_norm = grad.norm()
            if grad_norm > 1.:
                grad.div_(grad_norm)
            return grad


        # Compute grad norm and update discriminator
        for d_param, theta_0_param, theta_1_param in zip(D.parameters(), theta_0.parameters(), theta_2.parameters()):
            if theta_1_param.grad is not None:
                grad = theta_0_param.grad + theta_1_param.grad

                # simulate regularization with weight decay
                # if disc_reg > 0:
                #     grad += disc_reg * d_param.data

                # normalize gradient
                #grad = normalize_grad(grad)

                d_param.grad = 0.5 * grad

        for g_param, phi_0_param, phi_1_param in zip(G.parameters(), phi_0.parameters(), phi_2.parameters()):
            if phi_1_param.grad is not None:
                grad = phi_0_param.grad + phi_1_param.grad

                # normalize gradient
                #grad = normalize_grad(grad)

                g_param.grad = 0.5 * grad

        del theta_0, theta_1, theta_2
        del phi_0, phi_1, phi_2
        del D_norm_grads

        return G, D, errD, errG, D_x, D_G_z1, D_G_z2, grad_norm.detach(), disc_grad_norm.detach()



    def rk4_ode_step(self):
        G = self.gan.generator
        D = self.gan.discriminator
        data = self.gan.x
        step_size = self.config.ode_step_size or 0.1
        disc_reg = 0.01
        # Compute first step of RK4

        theta_1_cache, phi_1_cache, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(G.net, D.net, data,
                                                                               detach_err=False,
                                                                               retain_graph=True)

        # Compute the L2 norm using the prior computation graph
        grad_norm = None  # errG
        for phi_0_param in self.gan.generator.parameters():
            if phi_0_param.grad is not None:
                if grad_norm is None:
                    grad_norm = phi_0_param.grad.square().sum()
                else:
                    grad_norm = grad_norm + phi_0_param.grad.square().sum()

        grad_norm = grad_norm.sqrt()

        # Preserve gradients for regularization in cache
        #D_norm_grads = torch.autograd.grad(grad_norm, self.gan.discriminator.parameters())
        D_norm_grads = []
        grad_norm = grad_norm.detach()

        # Compute norm of the gradients of the discriminator for logging
        disc_grad_norm = torch.tensor(0.0, device="cuda:0")
        for d_grad, in zip(D_norm_grads):
            # compute discriminator norm
            disc_grad_norm = disc_grad_norm + d_grad.detach().square().sum().sqrt()

        # Detach graph
        errD = errD.detach()
        errG = errG.detach()

        # preserve theta1, phi1 for next computation
        theta_1 = self.grad_clone(theta_1_cache)
        phi_1 = self.grad_clone(phi_1_cache)

        # Update theta and phi for second RK step]
        for d_param, theta_1_param in zip(D.parameters(), theta_1.parameters()):
            if theta_1_param.grad is not None:
                theta_1_param.data = d_param.data + (step_size * 0.5 * -theta_1_param.grad)

        for g_param, phi_1_param in zip(G.parameters(), phi_1.parameters()):
            if phi_1_param.grad is not None:
                phi_1_param.data = g_param.data + (step_size * 0.5 * -phi_1_param.grad)

        # Compute second step of RK 4
        theta_2_cache, phi_2_cache, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(phi_1, theta_1, data)

        # preserve theta2, phi2
        theta_2 = self.grad_clone(theta_2_cache)
        phi_2 = self.grad_clone(phi_2_cache)

        # Update theta and phi for third RK step]
        for d_param, theta_2_param in zip(D.parameters(), theta_2.parameters()):
            if theta_2_param.grad is not None:
                theta_2_param.data = d_param.data + (step_size * 0.5 * -theta_2_param.grad)

        for g_param, phi_2_param in zip(G.parameters(), phi_2.parameters()):
            if phi_2_param.grad is not None:
                phi_2_param.data = g_param.data + (step_size * 0.5 * -phi_2_param.grad)

        # Compute third step of RK 4
        theta_3_cache, phi_3_cache, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(phi_2, theta_2, data)

        # preserve theta3, phi3
        theta_3 = self.grad_clone(theta_3_cache)
        phi_3 = self.grad_clone(phi_3_cache)

        # Update theta and phi for fourth RK step]
        for d_param, theta_3_param in zip(D.parameters(), theta_3.parameters()):
            if theta_3_param.grad is not None:
                theta_3_param.data = d_param.data + (step_size * -theta_3_param.grad)

        for g_param, phi_3_param in zip(G.parameters(), phi_3.parameters()):
            if phi_3_param.grad is not None:
                phi_3_param.data = g_param.data + (step_size * -phi_3_param.grad)

        # Compute fourth step of RK 4
        theta_4, phi_4, errD, errG, D_x, D_G_z1, D_G_z2 = self.ode_gan_step(phi_3, theta_3, data)
        # Inplace normalizes gradient; if grad_norm > 1
        def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
            # normalize gradient
            grad_norm = grad.norm()
            grad.div_(grad_norm)
            return grad


        # Compute grad norm and update discriminator
        for d_param, theta_1_param, theta_2_param, theta_3_param, theta_4_param in zip(D.parameters(),
                                                                                       theta_1_cache.parameters(),
                                                                                       theta_2_cache.parameters(),
                                                                                       theta_3_cache.parameters(),
                                                                                       theta_4.parameters()):
            if theta_1_param.grad is not None:
                grad = (theta_1_param.grad + 2 * theta_2_param.grad + 2 * theta_3_param.grad + theta_4_param.grad)

                # simulate regularization with weight decay
                # if disc_reg > 0:
                #     grad += disc_reg * d_param.data

                # normalize gradient
                grad = normalize_grad(grad)

                d_param.data = d_param.data + (step_size / 6. * -(grad))

        for g_param, phi_1_param, phi_2_param, phi_3_param, phi_4_param in zip(G.parameters(),
                                                                               phi_1_cache.parameters(),
                                                                               phi_2_cache.parameters(),
                                                                               phi_3_cache.parameters(),
                                                                               phi_4.parameters()):
            if phi_1_param.grad is not None:
                grad = (phi_1_param.grad + 2 * phi_2_param.grad + 2 * phi_3_param.grad + phi_4_param.grad)

                # normalize gradient
                grad = normalize_grad(grad)

                g_param.data = g_param.data + (step_size / 6.0 * -(grad))

        # Regularization step
        for d_param, d_grad in zip(D.parameters(), D_norm_grads):
            if d_param.grad is not None:
                d_param.data = d_param.data - step_size * disc_reg * d_grad

        del theta_1, theta_1_cache, theta_2, theta_2_cache, theta_3, theta_3_cache, theta_4
        del phi_1, phi_1_cache, phi_2, phi_2_cache, phi_3, phi_3_cache, phi_4
        del D_norm_grads

        return G, D, errD, errG, D_x, D_G_z1, D_G_z2, grad_norm.detach(), disc_grad_norm.detach()



    def calculate_gradients(self, add_train_hooks=True, create_graph=False, first_step=False):
        self.optimizer.zero_grad()
        d_loss, g_loss = self.trainable_gan.forward_loss()
        self.gan.add_metric('d_loss', d_loss.mean())
        self.gan.add_metric('g_loss', g_loss.mean())
        if add_train_hooks:
            for hook in self.train_hooks:
                if not first_step or hook.config.first_ode_step_only != True:
                    loss = hook.forward(d_loss, g_loss)
                    if loss[0] is not None:
                        d_loss += loss[0]
                    if loss[1] is not None:
                        g_loss += loss[1]

        self.trainable_gan.set_generator_trainable(True)
        self.trainable_gan.set_discriminator_trainable(False)

        g_loss.mean().backward(retain_graph=True, create_graph=create_graph)

        self.trainable_gan.set_generator_trainable(False)
        self.trainable_gan.set_discriminator_trainable(True)

        d_loss.mean().backward(retain_graph=True)
        self.trainable_gan.set_generator_trainable(True)

        d_grads = [p.grad * self.ttur for p in self.trainable_gan.d_parameters()]
        g_grads = [p.grad for p in self.trainable_gan.g_parameters()]
        gnd = sum([p.grad.norm() for p in self.trainable_gan.d_parameters()])
        self.gan.add_metric("GNd", gnd)
        gng = sum([p.grad.norm() for p in self.trainable_gan.d_parameters()])
        self.gan.add_metric("GNg", gng)
        return d_grads, g_grads

    def print_metrics(self, step):
        metrics = self.gan.metrics()
        metric_values = self.output_variables(metrics)
        print(str(self.output_string(metrics) % tuple([step] + metric_values)))
        for value in metric_values:
            if math.isnan(value):
                print("NAN detected, exitting")
                exit()

