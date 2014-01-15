import cPickle
import pylab
import numpy as np

log_god = cPickle.load(open("sidechains.log.learn.god"))
log_adapt = cPickle.load(open("sidechains.log.learn.adaptive"))
log_godobj = cPickle.load(open("sidechains.log.learn.godobj"))

me_adapt = np.mean(log_adapt[:, 2, :] / log_adapt[0, 2, :], axis=1)
me_god = np.mean(log_god[:, 2, :] / log_god[0, 2, :], axis=1)
me_godobj = np.mean(log_godobj[:, 2, :] / log_godobj[0, 2, :], axis=1)
pylab.plot(me_adapt, label='adaptive')
pylab.plot(me_god, label='god')
pylab.plot(me_godobj, label='god objective')
pylab.legend()
pylab.savefig('sidechain-energy.png')
pylab.close()

step_adapt = np.mean(log_adapt[:, 4, :], axis=1)
step_god = np.mean(log_god[:, 4, :], axis=1)
step_godobj = np.mean(log_godobj[:, 4, :], axis=1)
pylab.plot(step_adapt, label='adaptive')
pylab.plot(step_god, label='god')
pylab.plot(step_godobj, label='god objective')
pylab.legend()
pylab.savefig('sidechain-step.png')
pylab.close()

dist_adapt = np.mean(log_adapt[:, 6, :], axis=1)
dist_god = np.mean(log_god[:, 6, :], axis=1)
dist_godobj = np.mean(log_godobj[:, 6, :], axis=1)
pylab.plot(dist_adapt, label='adaptive')
pylab.plot(dist_god, label='god')
pylab.plot(dist_godobj, label='god objective')
pylab.legend()
pylab.savefig('sidechain-dist.png')
pylab.close()
