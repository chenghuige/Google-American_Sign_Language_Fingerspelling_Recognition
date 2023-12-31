import os

if os.path.exists('/kaggle'):
  import lele.layers
  import lele.losses
  from lele.layers.layers import Embedding
  from lele.distributed import parallel
  import lele.nlp 
  from lele.util import * 
  from lele.ops import *
  from lele.samplers import *
  from lele.training import *
  from lele.apps import * 
else:
  import lele.layers
  import lele.losses

  from lele.ops import *
  from lele.samplers import *

  try:
    import lele.fastai 
  except Exception:
    pass

  from lele.util import * 
  from lele.training import *
  from lele.apps import * 

  from lele.layers.layers import Embedding

  from lele.distributed import parallel
  import lele.nlp 
