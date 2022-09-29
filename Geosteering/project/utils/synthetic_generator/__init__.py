import os

import clr

dll_dir = os.getcwd() + '/dlls'
if not os.path.isdir(dll_dir):
    dll_dir = os.getcwd() + '/project/dlls'

apiDll = dll_dir + '/Api.dll'
auto_correlation_Dll = dll_dir + '/AutoCorrelation.dll'

clr.AddReference(apiDll)
clr.AddReference(auto_correlation_Dll)