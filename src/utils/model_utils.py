import json
import aiohttp
from typing import Dict, Any
from config.settings import HUGGINGFACE_TOKEN, DEFAULT_CONFIG_FILENAME

async def download_config(model_id: str, hf_token: str = None, filename: str = DEFAULT_CONFIG_FILENAME) -> Dict[str, Any]:
    """
    Asynchronously download model configuration from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        hf_token: HuggingFace API token (optional, uses default if not provided)
        filename: Configuration filename
        
    Returns:
        Model configuration dictionary
    """
    url = f"https://huggingface.co/{model_id}/raw/main/{filename}"
    headers = {
        "Authorization": f"Bearer {hf_token or HUGGINGFACE_TOKEN}",
        "Accept": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    raise Exception(f"Model or file not found: {model_id}/{filename}")
                elif response.status == 401:
                    raise Exception("Unauthorized: Please check your Hugging Face token")
                elif response.status != 200:
                    raise Exception(f"Error downloading {filename} from model {model_id}. HTTP {response.status}")
                
                # Try to parse as JSON first
                try:
                    return await response.json()
                except:
                    # If JSON parsing fails, try to parse the text content
                    text_content = await response.text()
                    try:
                        return json.loads(text_content)
                    except:
                        raise Exception(f"Failed to parse response as JSON. Response content: {text_content[:200]}...")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error while accessing Hugging Face: {str(e)}")

def estimate_parameters(config: Dict[str, Any]) -> int:
    """
    Estimate the total number of parameters of a Transformer model based on its config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Total number of parameters
    """
    vocab_size = config.get("vocab_size")
    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    num_attention_heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    num_key_value_heads = config.get("num_key_value_heads")
    intermediate_size = config.get("intermediate_size")
    
    if None in [vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size]:
        raise ValueError("The config is missing one or more required parameters.")
    
    embed_params = vocab_size * hidden_size
    q_params = hidden_size * (num_attention_heads * head_dim)
    k_params = hidden_size * (num_key_value_heads * head_dim)
    v_params = hidden_size * (num_key_value_heads * head_dim)
    o_params = (num_attention_heads * head_dim) * hidden_size
    attn_params = q_params + k_params + v_params + o_params
    mlp_params = 2 * hidden_size * intermediate_size
    layer_params = attn_params + mlp_params
    total_params = embed_params + num_hidden_layers * layer_params
    return total_params

def estimate_vram(total_params: int, q_bits: int) -> float:
    """
    Estimate VRAM requirements for a model.
    
    Args:
        total_params: Total number of model parameters
        q_bits: Quantization bits (e.g., 16 for FP16)
        
    Returns:
        Estimated VRAM requirement in GB
    """
    P_in_billions = total_params / 1e9
    return P_in_billions * 4 * (q_bits / 32) * 1.2