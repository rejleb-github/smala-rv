import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np


def run_smala(Niter, true_state, obs, eps, alpha):
    smala = mcmc.Smala(true_state,obs, eps, alpha)
    chain = np.zeros((0,smala.state.Nvars))
    chainlogp = np.zeros(0)
    tries = 0
    for i in range(Niter):
        if(smala.step()):
            tries += 1
        chainlogp = np.append(chainlogp,smala.state.get_logp(obs))
        chain = np.append(chain,[smala.state.get_params()],axis=0)
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    return smala, chain, chainlogp


def run_alsmala(Niter, true_state, obs, eps, alpha, bern_a, bern_b):
    alsmala = mcmc.Alsmala(true_state,obs, eps, alpha)
    chain = np.zeros((0,alsmala.state.Nvars))
    chainlogp = np.zeros(0)
    tries = 0
    for i in range(Niter):
        if( (np.exp(-bern_a*(i)/Niter)) >np.random.uniform()):
            if(alsmala.step()):
                tries +=1
        else:
            if(alsmala.step_mala()):
                tries += 1
        chainlogp = np.append(chainlogp,alsmala.state.get_logp(obs))
        chain = np.append(chain,[alsmala.state.get_params()],axis=0)
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    return alsmala, chain, chainlogp

def create_obs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs
