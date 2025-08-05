Peer 1 Perfromance - 

==========================================================

💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
📢 Last peer sent sampler output to all peers for step 0
⏱️ [req_1754431273675_0] sampler_hook: 0.74ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 1)...
📤 Sent sampler_output for req_1754431273675_0 step 0 to nodeacee...
💓 Sent heartbeat | CPU 1.0% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step1_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step1_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=1, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 1
⏱️ [TIMING] Inference Data Processing: 0.13ms name=req_1754431273675_0_step1_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2930.74ms
✅ Received hidden state and residual for req_1754431273675_0 (step 1). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 1):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 1:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 1):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.50ms
⏱️ [req_1754431273675_0] pre_hook_processing: 4.35ms
💓 Sent heartbeat | CPU 4.6% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:28
============================================================

📋 Request: req_1754431273675_0
   Step 1:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 5.0% VRAM 0.1 GB → ACK 
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
📢 Last peer sent sampler output to all peers for step 1
⏱️ [req_1754431273675_0] sampler_hook: 0.32ms
📤 Sent sampler_output for req_1754431273675_0 step 1 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 2)...
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:33
============================================================

📋 Request: req_1754431273675_0
   Step 1:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📨 Gateway received message: 'req_1754431273675_0_step2_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step2_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=2, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 2
⏱️ [TIMING] Inference Data Processing: 0.21ms name=req_1754431273675_0_step2_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 3587.89ms
✅ Received hidden state and residual for req_1754431273675_0 (step 2). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 2):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 2:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 2):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 4.65ms
⏱️ [req_1754431273675_0] pre_hook_processing: 5.90ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 2
⏱️ [req_1754431273675_0] sampler_hook: 0.22ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 3)...
📤 Sent sampler_output for req_1754431273675_0 step 2 to nodeacee...
💓 Sent heartbeat | CPU 1.6% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step3_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step3_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=3, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 3
⏱️ [TIMING] Inference Data Processing: 0.49ms name=req_1754431273675_0_step3_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2812.07ms
✅ Received hidden state and residual for req_1754431273675_0 (step 3). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 3):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 3:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 3):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 4.96ms
⏱️ [req_1754431273675_0] pre_hook_processing: 6.43ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 1.3% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:38
============================================================

📋 Request: req_1754431273675_0
   Step 3:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📢 Last peer sent sampler output to all peers for step 3
⏱️ [req_1754431273675_0] sampler_hook: 0.21ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 4)...
📤 Sent sampler_output for req_1754431273675_0 step 3 to nodeacee...
💓 Sent heartbeat | CPU 1.1% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step4_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step4_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=4, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 4
⏱️ [TIMING] Inference Data Processing: 0.18ms name=req_1754431273675_0_step4_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2900.49ms
✅ Received hidden state and residual for req_1754431273675_0 (step 4). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 4):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 4:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 4):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 4.56ms
⏱️ [req_1754431273675_0] pre_hook_processing: 5.69ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 0.2% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 4
⏱️ [req_1754431273675_0] sampler_hook: 0.46ms
📤 Sent sampler_output for req_1754431273675_0 step 4 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 5)...

============================================================
📊 INFERENCE_CONTEXT @ 15:01:44
============================================================

📋 Request: req_1754431273675_0
   Step 4:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 0.3% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step5_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step5_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=5, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 5
⏱️ [TIMING] Inference Data Processing: 0.15ms name=req_1754431273675_0_step5_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 1904.87ms
✅ Received hidden state and residual for req_1754431273675_0 (step 5). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 5):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 5:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 5):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.81ms
⏱️ [req_1754431273675_0] pre_hook_processing: 4.84ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 5
⏱️ [req_1754431273675_0] sampler_hook: 0.36ms
📤 Sent sampler_output for req_1754431273675_0 step 5 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 6)...
💓 Sent heartbeat | CPU 4.6% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step6_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step6_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=6, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 6
⏱️ [TIMING] Inference Data Processing: 0.19ms name=req_1754431273675_0_step6_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 1962.97ms
✅ Received hidden state and residual for req_1754431273675_0 (step 6). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 6):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 6:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 6):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.32ms
⏱️ [req_1754431273675_0] pre_hook_processing: 7.47ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms

============================================================
📊 INFERENCE_CONTEXT @ 15:01:49
============================================================

📋 Request: req_1754431273675_0
   Step 6:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 6
⏱️ [req_1754431273675_0] sampler_hook: 0.62ms
📤 Sent sampler_output for req_1754431273675_0 step 6 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 7)...
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step7_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step7_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=7, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 7
⏱️ [TIMING] Inference Data Processing: 0.30ms name=req_1754431273675_0_step7_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 1906.00ms
✅ Received hidden state and residual for req_1754431273675_0 (step 7). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 7):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 7:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 7):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.17ms
⏱️ [req_1754431273675_0] pre_hook_processing: 6.47ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms

============================================================
📊 INFERENCE_CONTEXT @ 15:01:54
============================================================

📋 Request: req_1754431273675_0
   Step 7:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 7
⏱️ [req_1754431273675_0] sampler_hook: 0.58ms
📤 Sent sampler_output for req_1754431273675_0 step 7 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 8)...
📨 Gateway received message: 'req_1754431273675_0_step8_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step8_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=8, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 8
⏱️ [TIMING] Inference Data Processing: 0.18ms name=req_1754431273675_0_step8_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 1975.27ms
✅ Received hidden state and residual for req_1754431273675_0 (step 8). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 8):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 8:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 8):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.08ms
⏱️ [req_1754431273675_0] pre_hook_processing: 3.69ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 4.2% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 8
⏱️ [req_1754431273675_0] sampler_hook: 0.19ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 9)...
📤 Sent sampler_output for req_1754431273675_0 step 8 to nodeacee...
💓 Sent heartbeat | CPU 4.0% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:59
============================================================

📋 Request: req_1754431273675_0
   Step 8:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📨 Gateway received message: 'req_1754431273675_0_step9_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step9_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms 
📋 Parsed: request_id='req_1754431273675_0', step=9, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 9
⏱️ [TIMING] Inference Data Processing: 0.19ms name=req_1754431273675_0_step9_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2251.53ms
✅ Received hidden state and residual for req_1754431273675_0 (step 9). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 9):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 9:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 9):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.50ms
⏱️ [req_1754431273675_0] pre_hook_processing: 7.39ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 3.8% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 9
⏱️ [req_1754431273675_0] sampler_hook: 0.24ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 10)...
📤 Sent sampler_output for req_1754431273675_0 step 9 to nodeacee...
💓 Sent heartbeat | CPU 3.1% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:04
============================================================

📋 Request: req_1754431273675_0
   Step 9:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📨 Gateway received message: 'req_1754431273675_0_step10_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step10_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=10, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 10
⏱️ [TIMING] Inference Data Processing: 0.17ms name=req_1754431273675_0_step10_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2445.98ms
✅ Received hidden state and residual for req_1754431273675_0 (step 10). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 10):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 10:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 10):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 4.30ms
⏱️ [req_1754431273675_0] pre_hook_processing: 5.29ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 2.6% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 10
⏱️ [req_1754431273675_0] sampler_hook: 0.19ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 11)...
📤 Sent sampler_output for req_1754431273675_0 step 10 to nodeacee...
💓 Sent heartbeat | CPU 2.0% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step11_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step11_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=11, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 11
⏱️ [TIMING] Inference Data Processing: 0.19ms name=req_1754431273675_0_step11_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2743.45ms
✅ Received hidden state and residual for req_1754431273675_0 (step 11). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 11):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 11:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 11):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 4.58ms
⏱️ [req_1754431273675_0] pre_hook_processing: 5.84ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:09
============================================================

📋 Request: req_1754431273675_0
   Step 11:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📢 Last peer sent sampler output to all peers for step 11
⏱️ [req_1754431273675_0] sampler_hook: 0.19ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 12)...
📤 Sent sampler_output for req_1754431273675_0 step 11 to nodeacee...
💓 Sent heartbeat | CPU 1.3% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step12_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step12_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=12, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 12
⏱️ [TIMING] Inference Data Processing: 0.19ms name=req_1754431273675_0_step12_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2894.60ms
✅ Received hidden state and residual for req_1754431273675_0 (step 12). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 12):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 12:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 12):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.12ms
⏱️ [req_1754431273675_0] pre_hook_processing: 4.37ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 0.8% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 12
⏱️ [req_1754431273675_0] sampler_hook: 0.33ms
📤 Sent sampler_output for req_1754431273675_0 step 12 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 13)...

============================================================
📊 INFERENCE_CONTEXT @ 15:02:15
============================================================

📋 Request: req_1754431273675_0
   Step 12:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step13_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step13_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=13, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 13
⏱️ [TIMING] Inference Data Processing: 0.14ms name=req_1754431273675_0_step13_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2067.35ms
✅ Received hidden state and residual for req_1754431273675_0 (step 13). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 13):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 13:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 13):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.43ms
⏱️ [req_1754431273675_0] pre_hook_processing: 5.10ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 13
⏱️ [req_1754431273675_0] sampler_hook: 0.56ms
📤 Sent sampler_output for req_1754431273675_0 step 13 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 14)...
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step14_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step14_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=14, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 14
⏱️ [TIMING] Inference Data Processing: 0.17ms name=req_1754431273675_0_step14_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 1920.95ms
✅ Received hidden state and residual for req_1754431273675_0 (step 14). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 14):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 14:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 14):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.63ms
⏱️ [req_1754431273675_0] pre_hook_processing: 6.69ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms

============================================================
📊 INFERENCE_CONTEXT @ 15:02:20
============================================================

📋 Request: req_1754431273675_0
   Step 14:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 14
⏱️ [req_1754431273675_0] sampler_hook: 0.42ms
📤 Sent sampler_output for req_1754431273675_0 step 14 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 15)...
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step15_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step15_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=15, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 15
⏱️ [TIMING] Inference Data Processing: 0.13ms name=req_1754431273675_0_step15_combined, tensor_shape=torch.Size([2, 1, 2048])⏱️ [req_17 754431273675_0] pre_hook_wait: 2129.11ms

✅ Received hidden state and residual for req_1754431273675_0 (step 15). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 15):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 15:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 15):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 3.93ms
⏱️ [req_1754431273675_0] pre_hook_processing: 4.49ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms

============================================================
📊 INFERENCE_CONTEXT @ 15:02:25
============================================================

📋 Request: req_1754431273675_0
   Step 15:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 15
⏱️ [req_1754431273675_0] sampler_hook: 0.49ms
📤 Sent sampler_output for req_1754431273675_0 step 15 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 16)...
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step16_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step16_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=16, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 16
⏱️ [TIMING] Inference Data Processing: 0.27ms name=req_1754431273675_0_step16_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 2148.41ms
✅ Received hidden state and residual for req_1754431273675_0 (step 16). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 16):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 16:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 16):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 7.78ms
⏱️ [req_1754431273675_0] pre_hook_processing: 9.22ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:30
============================================================

📋 Request: req_1754431273675_0
   Step 16:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📢 Last peer sent sampler output to all peers for step 16
⏱️ [req_1754431273675_0] sampler_hook: 0.89ms
📤 Sent sampler_output for req_1754431273675_0 step 16 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 17)...
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step17_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step17_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=17, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 17
⏱️ [TIMING] Inference Data Processing: 0.26ms name=req_1754431273675_0_step17_combined, tensor_shape=torch.Size([2, 1, 2048])
⏱️ [req_1754431273675_0] pre_hook_wait: 3200.75ms
✅ Received hidden state and residual for req_1754431273675_0 (step 17). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 17):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 17:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 17):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.16ms
⏱️ [req_1754431273675_0] pre_hook_processing: 6.46ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 0.1% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 17
⏱️ [req_1754431273675_0] sampler_hook: 0.20ms
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 18)...

============================================================
📊 INFERENCE_CONTEXT @ 15:02:36
============================================================

📋 Request: req_1754431273675_0
   Step 17:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📤 Sent sampler_output for req_1754431273675_0 step 17 to nodeacee...
💓 Sent heartbeat | CPU 2.9% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step18_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step18_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=18, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 18
⏱️ [TIMING] Inference Data Processing: 0.29ms name=req_1754431273675_0_step18_combined, tensor_shape=torch.Size([2, 1, 2048])⏱️ [req_17 754431273675_0] pre_hook_wait: 4544.04ms
✅ Received hidden state and residual for req_1754431273675_0 (step 18). Injecting into the next layer.

🔍 Raw tensor shapes before processing (step 18):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 18:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 18):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 5.61ms
⏱️ [req_1754431273675_0] pre_hook_processing: 6.98ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:41
============================================================

📋 Request: req_1754431273675_0
   Step 18:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📢 Last peer sent sampler output to all peers for step 18
⏱️ [req_1754431273675_0] sampler_hook: 0.49ms
📤 Sent sampler_output for req_1754431273675_0 step 18 to nodeacee...
🔍 Detected a non-first peer, waiting for hidden state from previous peer (step 19)...
💓 Sent heartbeat | CPU 4.7% VRAM 0.2 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:46
============================================================

📋 Request: req_1754431273675_0
   Step 18:
     - hidden_state: tensor torch.Size([1, 2048])
     - residual: tensor torch.Size([1, 2048])
============================================================

📨 Gateway received message: 'req_1754431273675_0_step19_combined' (tensor shape: torch.Size([2, 1, 2048]))
📊 Processing inference data: req_1754431273675_0_step19_combined
⏱️ [TIMING] Parse Tensor Metadata: 0.02ms
📋 Parsed: request_id='req_1754431273675_0', step=19, type='combined'
✅ Stored both hidden_state and residual for req_1754431273675_0 step 19
⏱️ [TIMING] Inference Data Processing: 0.28ms name=req_1754431273675_0_step19_combined, tensor_shape=torch.Size([2, 1, 2048])⏱️ [req_17 754431273675_0] pre_hook_wait: 3672.25ms

✅ Received hidden state and residual for req_1754431273675_0 (step 19). Injecting into the next layer.
🔍 Raw tensor shapes before processing (step 19):
   Original positions: torch.Size([1])
   Received hidden states: torch.Size([1, 2048])
   Received residual: torch.Size([1, 2048])
Dimensions of the hidden states: 2
Dimensions of the residual: 2
Dimensions of the positions: 1
🔍 Shape validation for step 19:
   Hidden states: torch.Size([1, 2048]) (expected seq_len: 1)
   Positions: torch.Size([1])
   Batch size: 1, Seq len: 1
🔧 Added batch dimension to hidden states: torch.Size([1, 1, 2048])
🔧 Added batch dimension to residual: torch.Size([1, 1, 2048])
🔧 Added batch dimension to positions: torch.Size([1, 1])
🔍 Normalized shapes:
   Hidden states: torch.Size([1, 1, 2048]) (batch=1, seq=1)
   Positions: torch.Size([1, 1]) (batch=1, seq=1)
🔧 Using last position for single token: torch.Size([1, 1])
✅ Final tensor shapes (step 19):
   Positions: torch.Size([1, 1])
   Hidden states: torch.Size([1, 1, 2048])
   Residual: torch.Size([1, 1, 2048])
⏱️ [req_1754431273675_0] tensor_manipulation: 2.84ms
⏱️ [req_1754431273675_0] pre_hook_processing: 4.10ms
⏱️ [req_1754431273675_0] post_hook_processing: 0.00ms
💓 Sent heartbeat | CPU 2.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 4.9% VRAM 0.2 GB → ACK 
📢 Last peer sent sampler output to all peers for step 19
⏱️ [req_1754431273675_0] sampler_hook: 0.22ms
Processed prompts: 100%|█████████████████████████████| 1/1 [01:36<00:00, 96.40s/it, est. speed input: 0.08 toks/s, output: 0.21 toks/s]
⏱️ [req_1754431273675_0] vllm_generate: 96422.25ms
🎯 Final result sent for req_1754431273675_0
🎉 Inference run completed for req_1754431273675_0

================================================================================
📊 PERFORMANCE PROFILE SUMMARY - req_1754431273675_0
================================================================================
🚀 vLLM Generate          : avg=96422.25ms max=96422.25ms min=96422.25ms total=96422.25ms count=1
⏳ Pre-hook Wait          : avg=2787.11ms max=5743.48ms min=1904.87ms total=55742.21ms count=20
🔄 Pre-hook Processing    : avg=   5.67ms max=   9.22ms min=   2.54ms total=  113.31ms count=20
📤 Post-hook Processing   : avg=   0.00ms max=   0.00ms min=   0.00ms total=    0.03ms count=20
📡 Tensor Send: No data
🎯 Sampler Hook           : avg=   0.40ms max=   0.89ms min=   0.19ms total=    7.93ms count=20
🧠 Layer Forward: No data
🔧 Tensor Manipulation    : avg=   4.45ms max=   7.78ms min=   2.12ms total=   89.06ms count=20
📊 Total Step: No data

🔥 TOP BOTTLENECKS:
   1. vLLM Generate: 96422.25ms (63.3%)
   2. Pre-hook Wait: 55742.21ms (36.6%)
   3. Pre-hook Processing: 113.31ms (0.1%)
================================================================================

🧹 Cleaned up context for req_1754431273675_0
🧹 Cleaned up step events for req_1754431273675_0
🧹 Cleaned up profile metrics for req_1754431273675_0
🔍 Output object type: <class 'str'>
📤 Sent sampler_output for req_1754431273675_0 step 19 to nodeacee...
✅ Sent final result to server for request req_1754431273675_0


Peer 2 - 
(tandemn-vllm) chopr@MSI:/mnt/c/Users/chopr/tandemn/tandemn-vllm$ python -m src.machine_runner
INFO 08-05 14:59:32 [__init__.py:244] Automatically detected platform cuda.
🔧 FORCED vLLM v0 mode (VLLM_USE_V1=0) for selective layer loading compatibility
2025-08-05T21:59:33.631712Z ERROR tensor_protocol: 🎉 [GET_ADDR] Created NodeTicket: nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae
🪪 TensorTransport started – ticket:
nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae

🪪 TensorTransport started – ticket:
nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae

🤖 Running as peer: nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae
✅ Registered in MongoDB as nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae
Starting unified message gateway...
🌐 Starting UNIFIED Message Gateway...
📊 Starting INFERENCE_CONTEXT debug monitor...
🔗 Central server ticket: nodeacc77kntmembpsj7qybhqtnekray42r66htp7m2t4hidv5gnkoskiaabacwbaapz4loqe
💓 Sent heartbeat | CPU 1.8% VRAM 7.7 GB → ACK
⏳ Waiting to be included in the pipeline...
🔗 Pipeline: ['nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae', 'nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae']
✅ Included in pipeline as nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae
✅ Position: 1 | First: False | Last: True
💓 Sent heartbeat | CPU 1.5% VRAM 7.7 GB → ACK 
💓 Sent heartbeat | CPU 2.2% VRAM 7.7 GB → ACK 
💓 Sent heartbeat | CPU 2.3% VRAM 7.6 GB → ACK 
/mnt/c/Users/chopr/tandemn/tandemn-vllm/src/utils/tensor_protocol_adapter.py:98: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
  return {"name": name, "tensor": torch.from_numpy(arr)}
📨 Gateway received message: 'deploy' (tensor shape: torch.Size([1379]))

================================================================================
📨 DEPLOYMENT MESSAGE RECEIVED
🔍 Action: deploy_model
ℹ️  Model: meta-llama/Llama-3.2-1B-Instruct, Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
================================================================================

🚀 Starting model deployment orchestration...
   Model: meta-llama/Llama-3.2-1B-Instruct
   Assigned layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
   Is first peer: True
   Is last peer: False
   Required files: 18
🔄 Deployment attempt 1/3
📥 Phase 1: Downloading model files...
📥 Starting download of 18 files...
📥 Downloading config/config.json...
✅ File already exists with correct size, skipping: config.json (872 bytes)
📥 Downloading config/tokenizer.json...
✅ File already exists with correct size, skipping: tokenizer.json (17,209,920 bytes)
📥 Downloading config/tokenizer_config.json...
✅ File already exists with correct size, skipping: tokenizer_config.json (50,521 bytes)
📥 Downloading embedding/layer.safetensors...
✅ File already exists with correct size, skipping: layer.safetensors (525,336,688 bytes)
📥 Downloading layers/layer_0.safetensors...
✅ File already exists with correct size, skipping: layer_0.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_1.safetensors...
✅ File already exists with correct size, skipping: layer_1.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_2.safetensors...
✅ File already exists with correct size, skipping: layer_2.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_3.safetensors...
💓 Sent heartbeat | CPU 2.3% VRAM 7.6 GB → ACK 
✅ File already exists with correct size, skipping: layer_3.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_4.safetensors...
✅ File already exists with correct size, skipping: layer_4.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_5.safetensors...
✅ File already exists with correct size, skipping: layer_5.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_6.safetensors...
✅ File already exists with correct size, skipping: layer_6.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_7.safetensors...
✅ File already exists with correct size, skipping: layer_7.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_8.safetensors...
✅ File already exists with correct size, skipping: layer_8.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_9.safetensors...
✅ File already exists with correct size, skipping: layer_9.safetensors (121,643,688 bytes)
📥 Downloading layers/layer_10.safetensors...
✅ File already exists with correct size, skipping: layer_10.safetensors (121,643,696 bytes)
📥 Downloading layers/layer_11.safetensors...
✅ File already exists with correct size, skipping: layer_11.safetensors (121,643,696 bytes)
📥 Downloading layers/layer_12.safetensors...
✅ File already exists with correct size, skipping: layer_12.safetensors (121,643,696 bytes)
📥 Downloading layers/layer_13.safetensors...
✅ File already exists with correct size, skipping: layer_13.safetensors (121,643,696 bytes)
✅ All 18 files downloaded successfully
🔧 Phase 2: Loading model with selective layers...
🔧 Loading model with selective layers...
Loading only a partial model for vLLM Inference
💓 Sent heartbeat | CPU 19.6% VRAM 7.6 GB → ACK 
💓 Sent heartbeat | CPU 10.7% VRAM 7.6 GB → ACK 
💓 Sent heartbeat | CPU 10.7% VRAM 7.6 GB → ACK 
INFO 08-05 14:59:50 [config.py:823] This model supports multiple tasks: {'classify', 'generate', 'score', 'reward', 'embed'}. Defaulting to 'generate'.
WARNING 08-05 14:59:50 [config.py:3271] Casting torch.bfloat16 to torch.float16.
INFO 08-05 14:59:50 [config.py:2195] Chunked prefill is enabled with max_num_batched_tokens=1024.
WARNING 08-05 14:59:50 [cuda.py:91] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
INFO 08-05 14:59:50 [llm_engine.py:230] Initializing a V0 LLM engine (v0.9.1) with config: model='deployed_models/meta-llama/Llama-3.2-1B-Instruct/config', speculative_config=None, tokenizer='deployed_models/meta-llama/Llama-3.2-1B-Instruct/config', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=128, download_dir=None, load_format=LoadFormat.DUMMY, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=None, served_model_name=deployed_models/meta-llama/Llama-3.2-1B-Instruct/config, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":0,"local_cache_dir":null}, use_cached_outputs=False,
WARNING 08-05 14:59:51 [interface.py:376] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
INFO 08-05 14:59:51 [cuda.py:327] Using Flash Attention backend.
💓 Sent heartbeat | CPU 9.3% VRAM 7.6 GB → ACK 
💓 Sent heartbeat | CPU 0.8% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 0.7% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 0.8% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 0.9% VRAM 7.4 GB → ACK 
[W805 15:00:01.462204852 socket.cpp:200] [c10d] The hostname of the client socket cannot be retrieved. err=-3
💓 Sent heartbeat | CPU 1.3% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 7.4 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 7.4 GB → ACK 
[W805 15:00:11.473691664 socket.cpp:200] [c10d] The hostname of the client socket cannot be retrieved. err=-3
INFO 08-05 15:00:12 [parallel_state.py:1065] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-05 15:00:12 [model_runner.py:1171] Starting to load model deployed_models/meta-llama/Llama-3.2-1B-Instruct/config...
💓 Sent heartbeat | CPU 3.8% VRAM 6.4 GB → ACK 
  Created REAL layer 0
  Created REAL layer 1
  Created REAL layer 2
  Created REAL layer 3
  Created REAL layer 4
  Created REAL layer 5
  Created REAL layer 6
  Created REAL layer 7
  Created REAL layer 8
  Created REAL layer 9
  Created REAL layer 10
  Created REAL layer 11
  Created REAL layer 12
  Created REAL layer 13
  Created PPMissingLayer for layer 14
  Created PPMissingLayer for layer 15
INFO 08-05 15:00:13 [model_runner.py:1203] Model loading took 2.0920 GiB and 0.689337 seconds
💓 Sent heartbeat | CPU 16.4% VRAM 0.1 GB → ACK 
INFO 08-05 15:00:14 [worker.py:294] Memory profiling takes 1.14 seconds
INFO 08-05 15:00:14 [worker.py:294] the current vLLM instance can use total_gpu_memory (8.00GiB) x gpu_memory_utilization (0.80) = 6.40GiB
INFO 08-05 15:00:14 [worker.py:294] model weights take 2.09GiB; non_torch_memory takes 0.03GiB; PyTorch activation peak memory takes 1.18GiB; the rest of the memory reserved for KV Cache is 3.10GiB.
INFO 08-05 15:00:15 [executor_base.py:113] # cuda blocks: 6340, # CPU blocks: 8192
INFO 08-05 15:00:15 [executor_base.py:118] Maximum concurrency for 128 tokens per request: 792.50x
💓 Sent heartbeat | CPU 55.6% VRAM 0.1 GB → ACK 
INFO 08-05 15:00:17 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 3.49 seconds
✅ Successfully created vLLM model with selective layers!
   Our monkey-patch created real layers for: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
   All other layers are PPMissingLayer (passthrough)
✅ Model loaded successfully!
✅ Model deployment orchestration completed successfully!
   Peer role: First
   Loaded layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
   Memory optimization: ~50.0% VRAM savings
✅ Model loaded successfully, registering inference hooks...
🔍 [DEBUG] Deployment status set to ready
📤 Reported deployment completion: {'model_name': 'meta-llama/Llama-3.2-1B-Instruct', 'peer_id': 'nodeaceehdriijydlc4at6udmd7k7dkpbcp65weyjfhophahsldshi3toaabacwbaapzq3zae', 'success': True}
💓 Sent heartbeat | CPU 3.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.0% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.0% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.1% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 1.3% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.8% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.9% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.5% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.3% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.3% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'inference' (tensor shape: torch.Size([631]))
⏱️ [TIMING] Parse Inference Trigger: 0.42ms 

================================================================================
📨 INFERENCE_TRIGGER MESSAGE RECEIVED
🔍 Action: start_inference
ℹ️  Request ID: req_1754431273675_0, Input: Hello, how are you today?...
================================================================================

📍 This peer is FIRST in the pipeline
✅ Starting inference as FIRST peer
🏃 Starting inference run in background thread...
✅ Inference started for request req_1754431273675_0✅ Dynamically attaching hooks to layers: LlamaDecoderLayer -> LlamaDecoderLayer   

⏱️ [TIMING] Inference Trigger Handling: 2.25ms tensor_size=<built-in method size of Tensor object at 0x7b4793d7f890>Starting the infereence run...

Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 79.25it/s]
Processed prompts:   0%|                                     | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
============================================================
📊 INFERENCE_CONTEXT @ 15:01:14
============================================================

📋 Request: req_1754431273675_0
============================================================

💓 Sent heartbeat | CPU 9.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 7.2% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:19
============================================================

📋 Request: req_1754431273675_0
============================================================

💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_0
⏱️ [req_1754431273675_0] post_hook_processing: 0.09ms
↪️ POST-HOOK for req_1754431273675_0, step 0: Sending hidden states...
🔍 Output tensor shapes before processing (step 0):
   Hidden states: torch.Size([8, 2048])
   Residual: torch.Size([8, 2048])
🔧 Step 0: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.46ms
⏳ Non-last peer waiting for sampler output for step 0
💓 Sent heartbeat | CPU 0.8% VRAM 0.1 GB → ACK 
📤 Sent combined tensors for req_1754431273675_0 step 0 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:24
============================================================

📋 Request: req_1754431273675_0
============================================================

📨 Gateway received message: 'req_1754431273675_0_step0_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step0_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.02ms
📋 Parsed: request_id='req_1754431273675_0', step=0, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 0
⏱️ [TIMING] Inference Data Processing: 0.55ms name=req_1754431273675_0_step0_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 0
⏱️ [req_1754431273675_0] sampler_hook: 5037.76ms
💓 Sent heartbeat | CPU 1.0% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_1
⏱️ [req_1754431273675_0] post_hook_processing: 0.06ms
↪️ POST-HOOK for req_1754431273675_0, step 1: Sending hidden states...
🔍 Output tensor shapes before processing (step 1):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 1: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 2.08ms
⏳ Non-last peer waiting for sampler output for step 1
📤 Sent combined tensors for req_1754431273675_0 step 1 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 4.7% VRAM 0.0 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:29
============================================================

📋 Request: req_1754431273675_0
   Step 0:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 5.0% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step1_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step1_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=1, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 1
⏱️ [TIMING] Inference Data Processing: 0.27ms name=req_1754431273675_0_step1_sampler_output, tensor_shape=torch.Size([250])✅ Received  sampler output for step 1

⏱️ [req_1754431273675_0] sampler_hook: 2684.40ms
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_2
⏱️ [req_1754431273675_0] post_hook_processing: 0.17ms
↪️ POST-HOOK for req_1754431273675_0, step 2: Sending hidden states...
🔍 Output tensor shapes before processing (step 2):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 2: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.80ms
⏳ Non-last peer waiting for sampler output for step 2
📤 Sent combined tensors for req_1754431273675_0 step 2 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 1.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:34
============================================================

📋 Request: req_1754431273675_0
   Step 1:
     - sampler_output: SamplerOutput
============================================================

📨 Gateway received message: 'req_1754431273675_0_step2_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step2_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=2, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 2
⏱️ [TIMING] Inference Data Processing: 0.24ms name=req_1754431273675_0_step2_sampler_output, tensor_shape=torch.Size([254])
✅ Received sampler output for step 2
⏱️ [req_1754431273675_0] sampler_hook: 2772.82ms
💓 Sent heartbeat | CPU 1.7% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_3
⏱️ [req_1754431273675_0] post_hook_processing: 0.06ms
↪️ POST-HOOK for req_1754431273675_0, step 3: Sending hidden states...
🔍 Output tensor shapes before processing (step 3):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 3: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.42ms
⏳ Non-last peer waiting for sampler output for step 3
📤 Sent combined tensors for req_1754431273675_0 step 3 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 1.5% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:40
============================================================

📋 Request: req_1754431273675_0
   Step 2:
     - sampler_output: SamplerOutput
============================================================

📨 Gateway received message: 'req_1754431273675_0_step3_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step3_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=3, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 3
⏱️ [TIMING] Inference Data Processing: 0.24ms name=req_1754431273675_0_step3_sampler_output, tensor_shape=torch.Size([254])✅ Received  sampler output for step 3

⏱️ [req_1754431273675_0] sampler_hook: 2859.76ms
💓 Sent heartbeat | CPU 1.3% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_4
⏱️ [req_1754431273675_0] post_hook_processing: 0.06ms
↪️ POST-HOOK for req_1754431273675_0, step 4: Sending hidden states...
🔍 Output tensor shapes before processing (step 4):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 4: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.41ms
⏳ Non-last peer waiting for sampler output for step 4
📤 Sent combined tensors for req_1754431273675_0 step 4 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 0.5% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step4_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step4_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=4, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 4
⏱️ [TIMING] Inference Data Processing: 0.19ms name=req_1754431273675_0_step4_sampler_output, tensor_shape=torch.Size([254])
✅ Received sampler output for step 4
⏱️ [req_1754431273675_0] sampler_hook: 3040.78ms
💓 Sent heartbeat | CPU 0.3% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:01:45
============================================================

📋 Request: req_1754431273675_0
   Step 4:
     - sampler_output: SamplerOutput
============================================================

🔍 Context key: sent_step_5
⏱️ [req_1754431273675_0] post_hook_processing: 0.24ms
↪️ POST-HOOK for req_1754431273675_0, step 5: Sending hidden states...
🔍 Output tensor shapes before processing (step 5):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 5: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.50ms
📤 Sent combined tensors for req_1754431273675_0 step 5 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 5
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step5_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step5_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.02ms
📋 Parsed: request_id='req_1754431273675_0', step=5, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 5
⏱️ [TIMING] Inference Data Processing: 0.23ms name=req_1754431273675_0_step5_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 5
⏱️ [req_1754431273675_0] sampler_hook: 1859.22ms
💓 Sent heartbeat | CPU 4.6% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_6
⏱️ [req_1754431273675_0] post_hook_processing: 0.34ms
↪️ POST-HOOK for req_1754431273675_0, step 6: Sending hidden states...
🔍 Output tensor shapes before processing (step 6):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 6: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.44ms
📤 Sent combined tensors for req_1754431273675_0 step 6 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 6

============================================================
📊 INFERENCE_CONTEXT @ 15:01:50
============================================================

📋 Request: req_1754431273675_0
   Step 5:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step6_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step6_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=6, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 6
⏱️ [TIMING] Inference Data Processing: 0.43ms name=req_1754431273675_0_step6_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 6
⏱️ [req_1754431273675_0] sampler_hook: 1843.69ms
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_7
⏱️ [req_1754431273675_0] post_hook_processing: 0.08ms
↪️ POST-HOOK for req_1754431273675_0, step 7: Sending hidden states...
🔍 Output tensor shapes before processing (step 7):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 7: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 12.89ms
📤 Sent combined tensors for req_1754431273675_0 step 7 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 7
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step7_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step7_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=7, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 7
⏱️ [TIMING] Inference Data Processing: 0.28ms name=req_1754431273675_0_step7_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 7
⏱️ [req_1754431273675_0] sampler_hook: 1920.32ms

============================================================
📊 INFERENCE_CONTEXT @ 15:01:55
============================================================

📋 Request: req_1754431273675_0
   Step 7:
     - sampler_output: SamplerOutput
============================================================

🔍 Context key: sent_step_8
⏱️ [req_1754431273675_0] post_hook_processing: 0.07ms
↪️ POST-HOOK for req_1754431273675_0, step 8: Sending hidden states...
🔍 Output tensor shapes before processing (step 8):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 8: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 3.19ms
⏳ Non-last peer waiting for sampler output for step 8
📤 Sent combined tensors for req_1754431273675_0 step 8 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 4.2% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step8_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step8_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=8, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 8
⏱️ [TIMING] Inference Data Processing: 0.32ms name=req_1754431273675_0_step8_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 8
⏱️ [req_1754431273675_0] sampler_hook: 2189.01ms
💓 Sent heartbeat | CPU 4.1% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_9
⏱️ [req_1754431273675_0] post_hook_processing: 0.07ms
↪️ POST-HOOK for req_1754431273675_0, step 9: Sending hidden states...
🔍 Output tensor shapes before processing (step 9):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 9: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 18.02ms
⏳ Non-last peer waiting for sampler output for step 9

============================================================
📊 INFERENCE_CONTEXT @ 15:02:00
============================================================

📋 Request: req_1754431273675_0
   Step 8:
     - sampler_output: SamplerOutput
============================================================

📤 Sent combined tensors for req_1754431273675_0 step 9 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 3.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step9_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step9_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=9, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 9
⏱️ [TIMING] Inference Data Processing: 0.32ms name=req_1754431273675_0_step9_sampler_output, tensor_shape=torch.Size([254])✅ Received  sampler output for step 9

⏱️ [req_1754431273675_0] sampler_hook: 2353.23ms
💓 Sent heartbeat | CPU 3.2% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_10
⏱️ [req_1754431273675_0] post_hook_processing: 0.08ms
↪️ POST-HOOK for req_1754431273675_0, step 10: Sending hidden states...
🔍 Output tensor shapes before processing (step 10):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 10: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 3.14ms
⏳ Non-last peer waiting for sampler output for step 10
📤 Sent combined tensors for req_1754431273675_0 step 10 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 2.7% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:05
============================================================

📋 Request: req_1754431273675_0
   Step 9:
     - sampler_output: SamplerOutput
============================================================

📨 Gateway received message: 'req_1754431273675_0_step10_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step10_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=10, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 10
⏱️ [TIMING] Inference Data Processing: 0.20ms name=req_1754431273675_0_step10_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 10
⏱️ [req_1754431273675_0] sampler_hook: 2701.67ms
💓 Sent heartbeat | CPU 2.3% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_11
⏱️ [req_1754431273675_0] post_hook_processing: 0.13ms
↪️ POST-HOOK for req_1754431273675_0, step 11: Sending hidden states...
🔍 Output tensor shapes before processing (step 11):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 11: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 1.74ms
⏳ Non-last peer waiting for sampler output for step 11
📤 Sent combined tensors for req_1754431273675_0 step 11 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 1.6% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:11
============================================================

📋 Request: req_1754431273675_0
   Step 10:
     - sampler_output: SamplerOutput
============================================================

📨 Gateway received message: 'req_1754431273675_0_step11_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step11_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=11, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 11
⏱️ [TIMING] Inference Data Processing: 0.29ms name=req_1754431273675_0_step11_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 11
⏱️ [req_1754431273675_0] sampler_hook: 2848.44ms
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_12
⏱️ [req_1754431273675_0] post_hook_processing: 0.07ms
↪️ POST-HOOK for req_1754431273675_0, step 12: Sending hidden states...
🔍 Output tensor shapes before processing (step 12):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 12: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 2.54ms
⏳ Non-last peer waiting for sampler output for step 12
📤 Sent combined tensors for req_1754431273675_0 step 12 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 0.8% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step12_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step12_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=12, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 12
⏱️ [TIMING] Inference Data Processing: 0.20ms name=req_1754431273675_0_step12_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 12
⏱️ [req_1754431273675_0] sampler_hook: 1945.39ms
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:16
============================================================

📋 Request: req_1754431273675_0
   Step 12:
     - sampler_output: SamplerOutput
============================================================

🔍 Context key: sent_step_13
⏱️ [req_1754431273675_0] post_hook_processing: 0.22ms
↪️ POST-HOOK for req_1754431273675_0, step 13: Sending hidden states...
🔍 Output tensor shapes before processing (step 13):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 13: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.78ms
📤 Sent combined tensors for req_1754431273675_0 step 13 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 13
💓 Sent heartbeat | CPU 4.6% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step13_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step13_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=13, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 13
⏱️ [TIMING] Inference Data Processing: 0.24ms name=req_1754431273675_0_step13_sampler_output, tensor_shape=torch.Size([254])
✅ Received sampler output for step 13
⏱️ [req_1754431273675_0] sampler_hook: 1945.67ms
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_14
⏱️ [req_1754431273675_0] post_hook_processing: 0.18ms
↪️ POST-HOOK for req_1754431273675_0, step 14: Sending hidden states...
🔍 Output tensor shapes before processing (step 14):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 14: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.46ms
📤 Sent combined tensors for req_1754431273675_0 step 14 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 14

============================================================
📊 INFERENCE_CONTEXT @ 15:02:21
============================================================

📋 Request: req_1754431273675_0
   Step 13:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step14_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step14_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=14, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 14
⏱️ [TIMING] Inference Data Processing: 0.16ms name=req_1754431273675_0_step14_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 14
⏱️ [req_1754431273675_0] sampler_hook: 1971.65ms
💓 Sent heartbeat | CPU 4.8% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_15
⏱️ [req_1754431273675_0] post_hook_processing: 0.34ms
↪️ POST-HOOK for req_1754431273675_0, step 15: Sending hidden states...
🔍 Output tensor shapes before processing (step 15):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 15: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.72ms
📤 Sent combined tensors for req_1754431273675_0 step 15 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 15
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step15_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step15_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=15, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 15
⏱️ [TIMING] Inference Data Processing: 0.30ms name=req_1754431273675_0_step15_sampler_output, tensor_shape=torch.Size([254])
✅ Received sampler output for step 15
⏱️ [req_1754431273675_0] sampler_hook: 1955.65ms

============================================================
📊 INFERENCE_CONTEXT @ 15:02:26
============================================================

📋 Request: req_1754431273675_0
   Step 15:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_16
⏱️ [req_1754431273675_0] post_hook_processing: 0.18ms
↪️ POST-HOOK for req_1754431273675_0, step 16: Sending hidden states...
🔍 Output tensor shapes before processing (step 16):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 16: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.64ms
📤 Sent combined tensors for req_1754431273675_0 step 16 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
⏳ Non-last peer waiting for sampler output for step 16
💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step16_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step16_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=16, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 16
⏱️ [TIMING] Inference Data Processing: 0.21ms name=req_1754431273675_0_step16_sampler_output, tensor_shape=torch.Size([250])✅ Receivedd sampler output for step 16
⏱️ [req_1754431273675_0] sampler_hook: 2302.58ms


============================================================
📊 INFERENCE_CONTEXT @ 15:02:31
============================================================

📋 Request: req_1754431273675_0
   Step 16:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_17
⏱️ [req_1754431273675_0] post_hook_processing: 0.09ms
↪️ POST-HOOK for req_1754431273675_0, step 17: Sending hidden states...
🔍 Output tensor shapes before processing (step 17):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 17: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.52ms
⏳ Non-last peer waiting for sampler output for step 17
📤 Sent combined tensors for req_1754431273675_0 step 17 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 0.1% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step17_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step17_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=17, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 17
⏱️ [TIMING] Inference Data Processing: 0.43ms name=req_1754431273675_0_step17_sampler_output, tensor_shape=torch.Size([254])
✅ Received sampler output for step 17
⏱️ [req_1754431273675_0] sampler_hook: 3071.00ms
💓 Sent heartbeat | CPU 2.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:36
============================================================

📋 Request: req_1754431273675_0
   Step 17:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.6% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_18
⏱️ [req_1754431273675_0] post_hook_processing: 0.09ms
↪️ POST-HOOK for req_1754431273675_0, step 18: Sending hidden states...
🔍 Output tensor shapes before processing (step 18):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 18: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 0.39ms
⏳ Non-last peer waiting for sampler output for step 18
📤 Sent combined tensors for req_1754431273675_0 step 18 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 1.4% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:42
============================================================

📋 Request: req_1754431273675_0
   Step 17:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.1 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step18_sampler_output' (tensor shape: torch.Size([254]))
📊 Processing inference data: req_1754431273675_0_step18_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=18, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 18
⏱️ [TIMING] Inference Data Processing: 0.42ms name=req_1754431273675_0_step18_sampler_output, tensor_shape=torch.Size([254])✅ Receivedd sampler output for step 18

⏱️ [req_1754431273675_0] sampler_hook: 3310.98ms
💓 Sent heartbeat | CPU 4.7% VRAM 0.1 GB → ACK 
🔍 Context key: sent_step_19
⏱️ [req_1754431273675_0] post_hook_processing: 0.12ms
↪️ POST-HOOK for req_1754431273675_0, step 19: Sending hidden states...
🔍 Output tensor shapes before processing (step 19):
   Hidden states: torch.Size([1, 2048])
   Residual: torch.Size([1, 2048])
🔧 Step 19: No slicing needed (full sequence or already single token).
⏱️ [req_1754431273675_0] tensor_send: 23.10ms
⏳ Non-last peer waiting for sampler output for step 19
📤 Sent combined tensors for req_1754431273675_0 step 19 to nodeaakes75zykufyhv44rmmlvbjylj6eyv2nez35rd2hhmrntduvmy5eaabacwbaapzupxae via TensorTransport
💓 Sent heartbeat | CPU 1.9% VRAM 0.1 GB → ACK 

============================================================
📊 INFERENCE_CONTEXT @ 15:02:47
============================================================

📋 Request: req_1754431273675_0
   Step 18:
     - sampler_output: SamplerOutput
============================================================

💓 Sent heartbeat | CPU 4.9% VRAM 0.2 GB → ACK 
📨 Gateway received message: 'req_1754431273675_0_step19_sampler_output' (tensor shape: torch.Size([250]))
📊 Processing inference data: req_1754431273675_0_step19_sampler_output
⏱️ [TIMING] Parse Tensor Metadata: 0.01ms
📋 Parsed: request_id='req_1754431273675_0', step=19, type='sampler_output'
✅ Stored sampler_output for req_1754431273675_0 step 19
⏱️ [TIMING] Inference Data Processing: 0.25ms name=req_1754431273675_0_step19_sampler_output, tensor_shape=torch.Size([250])
✅ Received sampler output for step 19
⏱️ [req_1754431273675_0] sampler_hook: 4838.85ms
Processed prompts: 100%|█████████████████████████████| 1/1 [01:36<00:00, 96.97s/it, est. speed input: 0.08 toks/s, output: 0.21 toks/s]
⏱️ [req_1754431273675_0] vllm_generate: 96991.39ms
🎉 Inference run completed for req_1754431273675_0

================================================================================
📊 PERFORMANCE PROFILE SUMMARY - req_1754431273675_0
================================================================================
🚀 vLLM Generate          : avg=96991.39ms max=96991.39ms min=96991.39ms total=96991.39ms count=1
⏳ Pre-hook Wait: No data
🔄 Pre-hook Processing: No data
📤 Post-hook Processing   : avg=   0.14ms max=   0.34ms min=   0.06ms total=    2.75ms count=20
📡 Tensor Send            : avg=   3.66ms max=  23.10ms min=   0.39ms total=   73.23ms count=20
🎯 Sampler Hook           : avg=2672.64ms max=5037.76ms min=1843.69ms total=53452.88ms count=20
🧠 Layer Forward: No data
🔧 Tensor Manipulation: No data
📊 Total Step: No data

🔥 TOP BOTTLENECKS:
   1. vLLM Generate: 96991.39ms (99.9%)
   2. Tensor Send: 73.23ms (0.1%)
================================================================================

🧹 Cleaned up context for req_1754431273675_0
🧹 Cleaned up step events for req_1754431273675_0
🧹 Cleaned up profile metrics for req_1754431273675_0
💓 Sent heartbeat | CPU 2.3% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.3% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.2% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.1% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.4% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.1% VRAM 0.1 GB → ACK 
💓 Sent heartbeat | CPU 0.1% VRAM 0.1 GB → ACK 



## 🔥 **CRITICAL BOTTLENECK ANALYSIS**

### **Peer 1 (Last Peer - Receives Tensors)**
```
🔥 TOP BOTTLENECKS:
   1. vLLM Generate: 96422.25ms (63.3%)  ← Expected, this is the actual inference
   2. Pre-hook Wait: 55742.21ms (36.6%)  ← 🚨 THIS IS YOUR KILLER
   3. Pre-hook Processing: 113.31ms (0.1%)
```

### **Peer 2 (First Peer - Sends Tensors)**  
```
🔥 TOP BOTTLENECKS:
   1. vLLM Generate: 96991.39ms (99.9%)  ← Expected
   2. Tensor Send: 73.23ms (0.1%)       ← Very fast, not the issue
```

## 🎯 **THE SMOKING GUN**

**Pre-hook Wait: 55,742ms (55.7 seconds!)** - This is Peer 1 waiting for tensors from Peer 2.

Looking at individual step timings:
- Step 1: `2930.74ms` wait
- Step 2: `3587.89ms` wait  
- Step 17: `4544.04ms` wait
- **Average: 2,787ms per step**

## 🔍 **ROOT CAUSE IDENTIFIED**

Your **token-by-token distributed communication** is creating a **massive serialization bottleneck**:

1. **Peer 2** processes one token → sends to Peer 1
2. **Peer 1** waits ~3 seconds for each token to arrive
3. This happens **20 times** (20 output tokens)
4. **Total waiting time: 55+ seconds**

## 🛠️ **WHAT TO OPTIMIZE (Priority Order)**

### **1. HIGHEST PRIORITY: Fix Communication Protocol**

**Problem**: Your current flow is:
```
Peer2: Process Token1 → Send → Wait
Peer1: Wait 3s → Receive → Process Token1 → Wait 3s → Receive Token2...
```

**Solution**: Implement **pipelined communication**:

```python
# In post_hook - instead of sending immediately, buffer tokens
def post_hook(module, args, output):
    # ... existing code ...
    
    # BUFFER MULTIPLE TOKENS BEFORE SENDING
    if not hasattr(hook_context, 'token_buffer'):
        hook_context['token_buffer'] = []
        hook_context['buffer_size'] = 4  # Send 4 tokens at once
    
    hidden_states, residual = output
    hook_context['token_buffer'].append((hidden_states, residual, current_step))
    
    # Send when buffer is full OR at end of sequence
    if len(hook_context['token_buffer']) >= hook_context['buffer_size'] or is_last_token:
        # Stack all buffered tokens
        batched_hidden = torch.stack([h for h, r, s in hook_context['token_buffer']])
        batched_residual = torch.stack([r for h, r, s in hook_context['token_buffer']])
        
        # Send batch with step indices
        step_indices = [s for h, r, s in hook_context['token_buffer']]
        
        with profile_timer("tensor_send_batch", request_id):
            await send_batched_tensors(batched_hidden, batched_residual, step_indices)
        
        hook_context['token_buffer'] = []
```

### **2. MEDIUM PRIORITY: Reduce Tensor Manipulation Overhead**

The tensor reshaping is taking ~4-5ms per step. You can optimize:

```python
# Pre-compute tensor shapes and reuse
def pre_hook(module, args):
    # Cache common tensor operations
    if not hasattr(hook_context, 'shape_cache'):
        hook_context['shape_cache'] = {}
    
    # Avoid repeated dimension checks
    if current_step in hook_context['shape_cache']:
        positions_to_inject, hidden_states_to_inject, residual_to_inject = hook_context['shape_cache'][current_step]
    else:
        # Do expensive tensor manipulation once
        # ... existing tensor manipulation code ...
        hook_context['shape_cache'][current_step] = (positions_to_inject, hidden_states_to_inject, residual_to_inject)
```

### **3. LOW PRIORITY: Network Optimization**

Tensor sending is already fast (73ms total), but you could:
- Use compression for tensor serialization
- Optimize the tensor transport protocol

## 📊 **EXPECTED IMPROVEMENTS**

If you implement batched communication (sending 4 tokens at once):

**Current**: 20 round-trips × 3 seconds = 60 seconds waiting
**Optimized**: 5 round-trips × 3 seconds = 15 seconds waiting

**Speedup**: ~4x reduction in communication overhead!

## 🚀 **IMMEDIATE ACTION ITEMS**

1. **Modify `post_hook`** to buffer tokens before sending
2. **Modify `pre_hook`** to handle batched tensor reception  
3. **Update tensor transport** to send/receive batches
4. **Test with different batch sizes** (2, 4, 8 tokens)

The data clearly shows your distributed communication protocol is the bottleneck, not vLLM itself. Fix the token-by-token serialization and you'll see massive speedups!
