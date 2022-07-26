# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk

from netket.driver import Fidopt

# 1D Lattice
L = 16
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# prepare ising GS
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=4.0)
ma = nk.models.RBM(alpha=1, param_dtype=complex)
sa = nk.sampler.MetropolisLocal(hi, n_chains=32)
op = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01)
vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=128)
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
gs.run(n_iter=100, out=None)


# the state we want to optimize
vs2 = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=128)
fo = Fidopt(
    vs2,
    vs,
    op,
    sr,
    logfid=False,
    fidbatchsize=128,
    nonzero_wavefun=True,
    apply_fun_is_logwf=True,
    reuse_target_samples=False,
)
fo.run(n_iter=300, out=None)
