# src/utils/tensor_transport.py
import asyncio
from typing import Any, Dict, Optional

# The wheel you built with build_pyo3_bindings.sh
import tensor_protocol as tp  


class TensorTransport:
    """
    Thin convenience wrapper around PyTensorNode.
    • Starts the QUIC node lazily.
    • Prints the self-ticket on first start (so you can copy-paste / log it).
    • Exposes send() / recv() coroutines that accept / return torch tensors.
    """
    # -------- life-cycle --------------------------------------------------

    def __init__(self) -> None:
        self._node: Optional[tp.PyTensorNode] = None
        self._ticket: Optional[str] = None          # Cached NodeAddr string

    async def start(self) -> None:
        if self._node is not None:       # already running
            return

        self._node = tp.PyTensorNode()
        await self._node.start()         # async start

        # Fetch the node-ticket (= shareable address string)
        self._ticket = await self._node.get_node_addr()
        print(f"🪪 TensorTransport started – ticket:\n{self._ticket}\n")

    async def stop(self) -> None:
        if self._node is not None:
            self._node.shutdown()
            self._node = None
            self._ticket = None

    # Optional helper so other modules can grab the ticket without awaiting
    @property
    def ticket(self) -> str:
        if self._ticket is None:
            raise RuntimeError("TensorTransport not started yet")
        return self._ticket

    # -------- data-plane --------------------------------------------------

    async def send(self, peer_addr: str, name: str, tensor) -> None:
        """
        peer_addr – ticket string of the remote peer
        name      – any identifier (not used by the protocol itself)
        tensor    – torch.Tensor or numpy.ndarray
        """
        import numpy as np                         # local import to avoid hard dep
        import torch

        if self._node is None:
            raise RuntimeError("TensorTransport.start() not called")

        if isinstance(tensor, torch.Tensor):
            array = tensor.cpu().contiguous().numpy()
        elif isinstance(tensor, np.ndarray):
            array = tensor
        else:
            raise TypeError("tensor must be torch.Tensor or numpy.ndarray")
        
        data = tp.PyTensorData(array.tobytes(), list(array.shape), str(array.dtype), False)
        await self._node.send_tensor(peer_addr, name, data)

    async def recv(self) -> Dict[str, Any]:
        """
        Blocks until *any* tensor arrives.
        Returns: {"name": str, "tensor": torch.Tensor}
        """
        import numpy as np
        import torch

        if self._node is None:
            raise RuntimeError("TensorTransport.start() not called")

        #TODO - Make SURE the PYO3 bindings are updated so that the 
        # name of the tensor is ALSO returned. The reason is that, 
        # it would be required to get it when we are serving MULTIPLE
        # requests at the same amount of time. 
        # name, pdata = await self._node.receive_tensor()
    
        pdata = await self._node.receive_tensor()
        if pdata is None:
            return None
        arr = np.frombuffer(pdata.as_bytes(), dtype=pdata.dtype).reshape(pdata.shape)
        # arr = np.frombuffer(pdata.data, dtype=pdata.dtype).reshape(pdata.shape)
        # return {"name": name, "tensor": torch.from_numpy(arr)}
        return {"tensor": torch.from_numpy(arr)}

    # -------- diagnostics -------------------------------------------------

    async def pool_size(self) -> int:
        if self._node is None:
            return 0
        return await self._node.pool_size()
