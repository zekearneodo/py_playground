import tensorflow as tf
from functional import compose, partial
import functools
from datetime import datetime
import os
import re
import sys
import numpy as np
import plot


def compose_all(*args):
    """Util for multiple function composition
    i.e. composed = composeAll([f, g, h])
         composed(x) == f(g(h(x)))
    """
    return partial(functools.reduce, compose)(*args)


def print_(var, name, first_n=5, summarize=5):
    """Util for debugging, by printing values of tf.Variable `var` during training"""
    # (tf.Tensor, str, int, int) -> tf.Tensor
    return tf.Print(var, [var], "{}: ".format(name), first_n=first_n,
                    summarize=summarize)


class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

    @staticmethod
    def wbVars(fan_in: int, fan_out: int):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))


class VAE():
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }

    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 save_graph_def=True, log_dir='./log'):
        """Build a symmetric VAE with given:
        end to end architecture [1000, 500, 250, 10]
        latent space 10, 1000 input, two hidden for each way
        """
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if not meta_graph:
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            # build the graph
            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        else:  # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_vae_")
            self.datetime = "{}_reloaded".format(model_datetime)
            *model_architecture, _ = re.split("_|-", model_name)
            self.architecture = [int(n) for n in model_architecture]

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        # unpack handles for tensor ops fed or fetch
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = handles

        if save_graph_def:  # tensorboard
            self.logger = tf.summary.FileWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        return self.global_step.eval(session=self.sesh)

    def _build_graph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name='x')
        dropout = tf.placeholder_with_default(1., shape=[], name='dropout')

        # encoding/recognition: q(z|x)
        encoding = [Dense("encoding", hidden_size, dropout, self.nonlinearity)
                    for hidden_size in self.architecture[::-1][1:-1]]
        h_encoded = compose_all(encoding)(x_in)

        # latent distribution parameterized by hidden encoding
        # z~N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense('z_mean', self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense('z_log_sigma', self.architecture[-1], dropout)(h_encoded)

        # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
        z = self.sample_gaussian(z_mean, z_log_sigma)

        # decoding/generation p(x|z)
        decoding = [Dense("dencoding", hidden_size, dropout, self.nonlinearity)
                    for hidden_size in self.architecture[1:-1]]
        # final reconstruction: restore original dims, squash outputs [0, 1]
        decoding.insert(0, Dense('x_decoding', self.architecture[0], dropout, self.squashing))
        x_reconstructed = tf.identity(compose_all(decoding)(z), name='x_reconstructed')

        # reconstruction loss: x vs x_reconstructed
        rec_loss = VAE.cross_entropy(x_reconstructed, x_in)

        # Kullback_Leibler divergence: x vs latent?
        kl_loss = VAE.kullback_leibler(z_mean, z_log_sigma)

        with tf.name_scope('l2_regularization'):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                'trainable_variables') if 'weights' in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope('cost'):
            # avg in minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name='vae_cost')
            cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            t_vars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, t_vars)
            for g, v in grads_and_vars:
                if g is None or v is None:
                    print('none in {} {}'.format(g, v))
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
                       for grad, tvar in grads_and_vars]
            print('clipped')
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name='minimize_cost')

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope('latent_space'):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                             shape=[None, self.architecture[-1]],
                                             name='latent_in')

        x_reconstructed_ = compose_all(decoding)(z_)

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_, x_reconstructed_, cost, global_step, train_op)

    def sample_gaussian(self, mu, log_sigma):
        with tf.name_scope('sample_gaussian'):
            epsilon = tf.random_normal(tf.shape(log_sigma), name='epsilon')
            return mu + epsilon * tf.exp(log_sigma)  # N(mu, I*sigma**2)

    @staticmethod
    def cross_entropy(obs, actual, offset=1e-7):
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope('cross_entropy'):
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope('l1_loss'):
            return tf.reduce_sum(tf.abs(obs - actual), 1)

    @staticmethod
    def l2_loss(obs, actual):
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope('l2_loss'):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullback_leibler(mu, log_sigma):
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        with tf.name_scope('KL_divergence'):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 -
                                        tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            # sample entered, have to update z_ in the graph with the entered sample
            is_tensor = lambda x: hasattr(x, 'eval')
            zs = (self.sesh.run(zs) if is_tensor(zs) else zs)  # coerec to np. array
            feed_dict.update({self.z_: zs})
        # else zs defaults to draw from conjugate prior z~N(0, I)
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        'end-to-end autoencoder'
        # np.array -> np.array
        return self.decode(self.sample_gaussian(*self.encode(x)))

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=True, out_dir='./out', plots_outdir='/.png',
              plot_latent_over_time=False):
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time:  # plot latent space over log_BASE time
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0

            while True:
                x, _ = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_: self.dropout}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i:
                        plot.explore_latent(self, nx=30, ny=30, ppf=True, out_dir=plots_outdir,
                                            name='explore_ppf30_{}'.format(pow_))
                        names = ('train', 'validation', 'test')
                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images, dataset.labels, range_=
                            (-6, 6), title=name, outdir=plots_out_dir,
                                              name="{}_{}".format(name, pow_))

                            print("{}^{} = {}".format(BASE, pow_, i))
                            pow_ += INCREMENT

                if i%1000 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i%2000 == 0 and verbose:
                    plot.plotSubset(self, x, x_reconstructed, n=10, name='train',
                                    outdir=plots_outdir)

                    if cross_validate:
                        x, _ = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)
                        print("round {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
                                        outdir=plots_outdir)

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))
                    if save:
                        out_file = os.path.join(os.path.abspath(out_dir), "{}_vae_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                        saver.save(self.sesh, out_file, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError):  # not logging
                        continue
                    break

        except KeyboardInterrupt:
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
