#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Logging utility module (in file in order to run on clusters).
# =============================================================================
import numpy as np
import os
from PIL import Image
import time

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.color import hsv2rgb
from skimage.draw import line

from yumi_push.common.misc import transparent_cmap

class Checkpoint(object):

    def __init__(self, config, tag=None):
        super(Checkpoint, self).__init__()
        # Initializing checkpoint module.
        self.ready = False
        self.config = config
        self.log = []
        self.can_reset = True
        # Building model directory based on name and time.
        now = time.strftime("%H_%M_%S_%d_%b", time.gmtime())
        tag = config["config"] + "_" + now if tag is None else tag
        self.dir = os.path.join(os.environ["YUMI_PUSH_OUTS"], tag)
        # Creating output directories for model, results, logging, config, etc.
        os.makedirs(self.dir, exist_ok=True)
        open_type = "a" if os.path.exists(self.get_path("logging.txt")) else "w"
        self.log_file = open(self.get_path("logging.txt"), open_type)
        with open(self.get_path("config.txt"), open_type) as f:
            for key, value in config.items():
                f.write("{}: {}\n".format(key, value))
            f.write("\n")
        self.debug = config.get("debug", False)
        os.makedirs(self.get_path("debug"), exist_ok=True)
        # Creating auxialiary variables.
        self.iteration = 0
        self.object_color_dict = {}
        self.verbose = config.get("verbose", False)
        # Logging variables.
        self._last_state = None
        self._seg_rews = []
        self._starting_pos = []
        self._logging_len = config.get("ckp_testing_interval", 200)
        self.is_logging_step = False
        self.is_testing_step = False
        # Set OPENAI logging directory.
        os.environ["OPENAI_LOGDIR"] = self.dir
        self.ready = True

    def step(self):
        self.can_reset = True
        self.iteration = int(self.iteration + 1)
        self.write_log("\nSTEP {}".format(self.iteration))

    def reset(self):
        self.save()
        if self.can_reset:
            self.log.append({"episode": len(self.log)+1})
            self.can_reset = False
        self.is_testing_step = (self.iteration % self._logging_len == 0)
        self.iteration += 0.5
        self.write_log("\nRESET - STEP {}".format(self.iteration))

    # =========================================================================
    # Saving and Plotting.
    # =========================================================================
    def save(self):
        batch_steps = self.config.get("ckp_plot_save_interval", 1000)
        if len(self.log) < batch_steps: return
        elif not len(self.log) % batch_steps==0: return
        log_values = {x: [] for x in self.log[-1].keys()}
        for x in log_values.keys():
            log_values[x] = []
            batch = []
            for logs in self.log:
                batch.append(logs[x])
                if len(batch) >= batch_steps:
                    log_values[x].append(np.mean(batch))
                    batch = []
        for label, values in log_values.items():
            if label == "episode": continue
            fig = plt.figure()
            episode = len(values)*batch_steps
            axis = np.linspace(1, episode, len(values))
            plt.title("{} curve".format(label))
            plt.plot(axis, values, label=str(label))
            plt.legend()
            plt.xlabel("EPISODES")
            plt.ylabel(str(label).upper())
            plt.grid(True)
            plt.savefig(self.get_path("log_{}.pdf".format(label)))
            plt.close(fig)

    def save_action_as_plot(self, start, target, image, name="action.png"):
        """ Draws an action in an image by drawing line from start to
        target position. Both are thereby assumed to be in pixel coordinates. """
        if not self.debug: return
        assert start.size == 2 and target.size == 2
        h,w = image.shape[:2]
        data = self.segmented_to_image(image)
        if not (0 <= start[0] <= h and 0 <= start[1] <= w): return
        if not (0 <= target[0] <= h and 0 <= target[1] <= w): return
        rr, cc = line(start[0], start[1], target[0], target[1])
        data[rr,cc,:] = 255
        name = "{}_{}".format(self.iteration, name)
        plt.imshow(data)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def save_action_dist(self, name="start_action_dis.png"):
        """ Plot distribution of starting positions over segmented image. """
        #if not self.debug: return
        if len(self._starting_pos) < self._logging_len-1: return
        segmented = self._last_state[1].copy()
        array = np.zeros(segmented.shape)
        for x in self._starting_pos: array[x[0],x[1]] += 1
        cmap_transparent = transparent_cmap(plt.cm.Reds)
        sns.heatmap(array,vmin=0,vmax=self._logging_len/10,cmap=cmap_transparent)
        plt.imshow(self.segmented_to_image(segmented), alpha=0.6)
        name = "{}_{}".format(self.iteration, name)
        title="Start Point Distribution - Iteration = {}".format(self.iteration)
        plt.title(title)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def save_array_as_plot(self, array, name="array.png"):
        """ Converts numpy array to image and saves it.
        Only active if debug flat (in config) is set to True. """
        if not self.debug: return
        assert len(array.shape) == 2 or len(array.shape) == 3
        data = array.copy()
        data[data < 0] = 0
        max_value = max(np.amax(array), 1e-4)
        data = (data*int(255.0/max_value)).astype(np.uint8)
        name = "{}_{}".format(self.iteration, name)
        plt.imshow(data)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def save_distances_as_plot(self, array, name="distances.png"):
        """ Converts distance array to heatmap and saves as plot. """
        if not self.debug: return
        assert len(array.shape) == 2 or len(array.shape) == 3
        sns.heatmap(array)
        name = "{}_{}".format(self.iteration, name)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def save_network_activation_as_plot(self, array, name="activations_sum"):
        """ Determine activation of first fully connected layer and plot them.
        For efficiency sum the activations of all layers to one plot. """
        #if not self.debug: return
        if "conv" in self.config["model_network"]: return
        if self.config["train_algo"] != "ppo2": return
        session = tf.get_default_session()
        #print([n.name for n in tf.get_default_graph().as_graph_def().node])
        layer = session.graph.get_tensor_by_name("ppo2_model/pi/fc1/w:0")
        placeholder = session.graph.get_tensor_by_name("ppo2_model/Ob:0")
        feed_data = np.reshape(array, [1,array.shape[0],array.shape[1],1])
        feed_data = feed_data.astype(np.float32)
        units = np.asarray(session.run(layer,feed_dict={placeholder:feed_data}))
        # Plot sum of activations over all neurons on first layer.
        units_summed = np.sum(units, axis=1)
        img_s = int(np.sqrt(units_summed.shape[0])) # assumes image to be quadratic.
        sns.heatmap(np.reshape(units_summed, (img_s, img_s)))
        name = "{}_{}.png".format(self.iteration, name)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()
        # Plot differences in activation.
        # ...

    def save_qmaps_as_plot(self, qmaps, directions, segmented=None):
        """ Plot qmaps (x,y,direction_index) as heatmaps for every direction. """
        #if not self.debug: return
        fig, ax = plt.subplots(1,len(directions),figsize=(10*len(directions),7))
        if not type(ax) == np.ndarray: ax = [ax]
        title = "QMaps - Iteration = {}".format(self.iteration)
        st = fig.suptitle(title, fontsize="x-large")
        max_reward = self.config.get("rew_final", 1000)
        for id,dir in enumerate(directions):
            array = qmaps[:,:,id]
            sns.heatmap(array, vmax=max_reward, ax=ax[id])
            ax[id].title.set_text("Direction={}".format(dir))
        name = "{}_qvalues.png".format(self.iteration)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()
        if segmented is not None:
            data = self.segmented_to_image(segmented)
            name = "{}_qvalues_segmented.png".format(self.iteration)
            plt.imshow(data)
            plt.savefig(os.path.join(self.dir, "debug", name))
            plt.close()

    def save_segmentation_as_plot(self, array, name="segment.png"):
        """ Segmented images cannot be visualised as they are, as the class
        indexes are very low, so that the image will mostly black and small
        difference between the classes. Therefore the classes are transformed
        to distinguishable colors and then the image is stored.
        Only active if debug flat (in config) is set to True. """
        if not self.debug: return
        data = self.segmented_to_image(array)
        name = "{}_{}".format(self.iteration, name)
        plt.imshow(data)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def save_seg_rew_map(self, name="seg_rew_map.png"):
        """ Summarize the list of "reward-weighted" segmentation maps
        and plot as array. """
        if not self.debug: return
        if len(self._seg_rews) < self._logging_len-1: return
        array = np.zeros((len(self._seg_rews), *self._seg_rews[0].shape))
        for i,x in enumerate(self._seg_rews): array[i,:,:] = x
        non_zero = np.count_nonzero(array, axis=0)
        array = np.divide(np.sum(array, axis=0), non_zero)
        sns.heatmap(array, vmin=0, vmax=np.nanmax(array))
        name = "{}_{}".format(self.iteration, name)
        plt.savefig(os.path.join(self.dir, "debug", name))
        plt.close()

    def segmented_to_image(self, array):
        assert len(array.shape) == 2
        classes = np.unique(array)
        if classes[-1] == 255: classes = classes[:-1]
        if classes[0] <= 0: classes = classes[1:]

        def random_color():
            return (np.array([0.286,0.4,0.64])*255).astype(np.uint8)
            #return (np.random.rand(1,3)*255).astype(np.uint8)

        colors = []
        for idx in classes:
            if not idx in self.object_color_dict.keys():
                self.object_color_dict[idx] = random_color()
            colors.append(self.object_color_dict[idx])
        array_flatten = array.flatten()
        data = np.zeros((array_flatten.size, 3))
        for i,c in enumerate(classes):
            data[array_flatten==c, :] = colors[i]
        data = np.reshape(data, (*array.shape, 3)).astype(np.uint8)
        return data

    # =========================================================================
    # Updating Functions.
    # =========================================================================
    def update_last_state(self, state, segmented, distances):
        self._last_state = (state, segmented, distances)

    def update_seg_rews(self, reward):
        if self._last_state is None: return []
        self._seg_rews.append(reward*self._last_state[1])
        if len(self._seg_rews) > self._logging_len: self._seg_rews[:1]=[]

    def update_start_pos(self, pstart_pos):
        self._starting_pos.append(pstart_pos)
        if len(self._starting_pos) > self._logging_len: self._starting_pos[:1]=[]

    # =========================================================================
    # Logging.
    # =========================================================================
    def add_log(self, log, desc, typ="set"):
        """ Add value to internal log, by expanding internal logging dictionary
        by the given description (if not already existing). Several kinds of
        storage are possible, i.e. "add", "set". """
        assert len(self.log) > 0
        if typ == "set":
            self.log[-1][desc] = log
            return
        elif typ == "add":
            if not desc in self.log[-1].keys(): self.log[-1][desc] = 0.0
            self.log[-1][desc] += log
            return
        else:
            raise ValueError("Invalid logging type {}".format(typ))

    def write_log(self, log, refresh=False):
        self.log_file.write(log + "\n")
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path("logging.txt"), "a")
        if self.verbose: print(log)

    def done(self):
        self.ready = False
        self.save()
        self.log_file.close()

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

# =============================================================================
# Timer class.
# =============================================================================
class Timer(object):

    def __init__(self):
        super(Timer, self).__init__()
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.reset()
        return ret

    def reset(self):
        self.acc = 0
