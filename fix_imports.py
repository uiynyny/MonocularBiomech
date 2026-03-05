import os
import glob

for f in glob.glob('/Users/dexz/workspace/MonocularBiomechanics/metrabs_tf/**/*.py', recursive=True):
    with open(f, 'r') as fp:
        c = fp.read()
    c = c.replace('from posepile.paths import DATA_ROOT', 'from metrabs_tf.paths import DATA_ROOT')
    c = c.replace('from posepile.joint_info import JointInfo', 'from metrabs_tf.posepile import JointInfo')
    c = c.replace('import posepile.datasets3d as ds3d', 'ds3d = None')
    c = c.replace('import posepile.datasets2d as ds2d', 'ds2d = None')
    with open(f, 'w') as fp:
        fp.write(c)
