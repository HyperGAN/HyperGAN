class BaseSampler:
    def __init__(self, gan, samples_per_row=8):
        self.gan = gan
        self.samples_per_row = samples_per_row

    def _sample(self):
        raise "raw _sample method called.  You must override this"

    def sample(self, path):
        if not self.gan.created:
            self.gan.create()

        sample = self._sample()

        data = sample['generator'] #TODO variable

        width = min(gan.config.batch_size, self.samples_per_row)
        stacks = [np.hstack(data[i*width:i*width+width]) for i in range(gan.config.batch_size//width)]
        sample_data = np.vstack(stacks)
        plot(config, sample_data, sample_file)
        sample_name = 'generator'
        samples = [[sample_data, sample_name]]

        return [{'image':sample_file, 'label':sample_name} for sample_data, sample_filename in samples]
