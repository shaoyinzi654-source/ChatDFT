import json
import re
from openai import OpenAI

# Initialize the OpenAI client with the provided credentials
client = OpenAI(
    api_key="sk-omWl5smTaCBOrRalgbbBk09Migy7e1w1J9raZXRDYjkzDfoY",
    base_url="https://tokken.cc/v1"
)

SYSTEM_PROMPT = """You are an expert quantum chemist and AI assistant. Your task is to translate a user's natural language request for a DFT (Density Functional Theory) or Hartree-Fock calculation into a structured JSON configuration that our calculation engines can run.

Our engines support highly complex simulations:

1. "1d_dft" (1D Kohn-Sham DFT solver on a grid):
   - This solver is extremely general. It can simulate ANY atomic chain, molecular array, periodic lattice, or trap system in 1D.
   - Params:
     * num_electrons: total electrons (e.g., 1 to 100).
     * L: half-width of grid (typically 10.0 to 15.0).
     * N: grid points (typically 200).
     * max_iter: max iterations (default 100).
     * tol: tolerance (default 1e-6).
     * alpha: density mixing rate (default 0.2).
     * softening: Coulomb softening factor 'a' (default 1.0).
     * functional: "LDA", "GGA-PBE", "Exchange-Only", or "Hartree".
     * mixing_method: "Linear" or "Anderson".
     * potential_expr: A Python math expression in terms of 'x'.
       You can construct:
       - Multi-atom 1D chain (e.g., alternating Li and H chain with 4 atoms at x = -4.5, -1.5, 1.5, 4.5):
         "-3.0 / np.sqrt((x + 4.5)**2 + softening**2) - 1.0 / np.sqrt((x + 1.5)**2 + softening**2) - 3.0 / np.sqrt((x - 1.5)**2 + softening**2) - 1.0 / np.sqrt((x - 4.5)**2 + softening**2)"
       - Lattice defect / Impurity in a periodic potential:
         "-3.0 * np.cos(2 * np.pi * x / 3.0)**2 - 2.0 * np.exp(-x**2)" (Periodic wells + localized defect well)
       - Molecule under electric field E:
         "-Z1 / np.sqrt((x - d/2)**2 + softening**2) - Z2 / np.sqrt((x + d/2)**2 + softening**2) + E * x"
     * potential_description: A human-readable Chinese description of the physical system.

2. "3d_diatomic" (3D Hartree-Fock / Molecular SCF solver using STO-3G basis set):
   - Supports arbitrary multi-atom systems (from diatomic to triatomic H2O, CO2, ammonia NH3, or clusters) for H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg.
   - Instead of separate atom fields, you must define the whole molecular geometry using the "atoms" list of dicts.
   - Params:
     * atoms: A list of dicts, where each dict has "name" (element name) and "pos" ([x, y, z] coordinates in Bohr).
       For example, Water (H2O):
       [
         {"name": "O", "pos": [0.0, 0.0, 0.12]},
         {"name": "H", "pos": [0.0, 1.43, -0.98]},
         {"name": "H", "pos": [0.0, -1.43, -0.98]}
       ]
       For CO2:
       [
         {"name": "C", "pos": [0.0, 0.0, 0.0]},
         {"name": "O", "pos": [0.0, 0.0, -2.19]},
         {"name": "O", "pos": [0.0, 0.0, 2.19]}
       ]
     * num_electrons: Total electrons of the molecular system (sum of atomic numbers, e.g. 10 for H2O, 22 for CO2, 14 for CO, 14 for N2, 10 for HF, 12 for LiF, etc.), up to 24 electrons.
     * max_iter: max iterations (default 50).
     * tol: tolerance (default 1e-6).

### LAUNCH OPTIMIZATION FLAG:
- If the user explicitly asks to "optimize the geometry", "relax the structure", "find the equilibrium bond length", "结构优化", "弛豫结构", or "寻找平衡键长" in 3D:
  Set the root-level key `"launch_optimization": true`. Otherwise set it to `false`.

### RESPONSE FORMAT:
You MUST respond ONLY with a valid JSON object matching the following structure, with no markdown styling or conversational text:
{
  "solver_type": "1d_dft" or "3d_diatomic",
  "explanation": "中文详细解释该计算体系的物理和化学模拟逻辑",
  "launch_optimization": true or false,
  "params": {
    // If solver_type is 1d_dft:
    "num_electrons": 4,
    "L": 12.0,
    "N": 200,
    "max_iter": 100,
    "tol": 1e-6,
    "alpha": 0.15,
    "softening": 1.0,
    "functional": "GGA-PBE",
    "mixing_method": "Anderson",
    "potential_expr": "-3.0 / np.sqrt((x+3)**2 + softening**2) - 1.0 / np.sqrt(x**2 + softening**2) - 3.0 / np.sqrt((x-3)**2 + softening**2)",
    "potential_description": "一维 Li-H-Li 分子链"
    
    // If solver_type is 3d_diatomic (multi-atom general):
    "atoms": [
      {"name": "O", "pos": [0.0, 0.0, 0.12]},
      {"name": "H", "pos": [0.0, 1.43, -0.98]},
      {"name": "H", "pos": [0.0, -1.43, -0.98]}
    ],
    "num_electrons": 10,
    "max_iter": 50,
    "tol": 1e-6
  }
}
"""

def parse_user_request(prompt: str, history: list = None, model_name: str = "gpt-5.6-luna") -> dict:
    """
    Calls the specified AI model to parse the user's natural language request into a DFT config JSON.
    """
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for item in history:
                # Add preceding user prompts and assistant answers to messages
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": f"User Request: {prompt}"})
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1
        )
        
        content = completion.choices[0].message.content.strip()
        
        # Strip markdown code blocks if the model included them
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()
            
        data = json.loads(content)
        return data
    except Exception as e:
        # Fallback if something fails
        print(f"Error calling AI parser with model {model_name}: {e}")
        return {
            "solver_type": "1d_dft",
            "explanation": f"模型 {model_name} 解析失败 ({str(e)})。已加载默认系统作为备份。",
            "launch_optimization": False,
            "params": {
                "num_electrons": 2,
                "L": 10.0,
                "N": 200,
                "max_iter": 100,
                "tol": 1e-6,
                "alpha": 0.2,
                "softening": 1.0,
                "functional": "LDA",
                "mixing_method": "Linear",
                "potential_expr": "0.5 * (x ** 2)",
                "potential_description": "一维谐振势阱 (AI备份)"
            }
        }
