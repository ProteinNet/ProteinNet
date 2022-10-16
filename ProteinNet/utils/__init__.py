#from utils.seed import set_seed
from ProteinNet.ProteinNet.utils.seed import set_seed
from ProteinNet.ProteinNet.preprocess.pyg_util import dense_to_sparse, to_dense_adj
#from ..preprocess.pyg_util import dense_to_sparse, to_ense_adj
from ProteinNet.ProteinNet.preprocess.ogb_util import get_atom_feature_dims, get_bond_feature_dims
#from utils.nt_xent import NTXentLoss
#from utils.pooling import TopKPooling
