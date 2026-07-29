import json
import os
import re

from openai import OpenAI


def _build_client():
    api_key = os.getenv("CHAT_DFT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("CHAT_DFT_BASE_URL", "https://tokken.cc/v1")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


client = _build_client()

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
     * potential_description: A human-readable Chinese description of the physical system.

2. "3d_diatomic" (3D Hartree-Fock / Molecular SCF solver using STO-3G basis set):
   - Supports arbitrary multi-atom systems (from diatomic to triatomic H2O, CO2, ammonia NH3, or clusters) for H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg.
   - Instead of separate atom fields, you must define the whole molecular geometry using the "atoms" list of dicts.
   - Params:
     * atoms: A list of dicts, where each dict has "name" (element name) and "pos" ([x, y, z] coordinates in Bohr).
     * num_electrons: Total electrons of the molecular system.
     * max_iter: max iterations (default 50).
     * tol: tolerance (default 1e-6).

### LAUNCH OPTIMIZATION FLAG:
- If the user explicitly asks to "optimize the geometry", "relax the structure", "find the equilibrium bond length", "结构优化", "弛豫结构", or "寻找平衡键长" in 3D:
  Set the root-level key "launch_optimization": true. Otherwise set it to false.

### RESPONSE FORMAT:
You MUST respond ONLY with a valid JSON object matching the following structure, with no markdown styling or conversational text:
{
  "solver_type": "1d_dft" or "3d_diatomic",
  "explanation": "中文详细解释该计算体系的物理和化学模拟逻辑",
  "launch_optimization": true or false,
  "params": {
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
  }
}
"""


def parse_user_request(prompt: str, history: list = None, model_name: str = "gpt-5.6-luna") -> dict:
    """Translate a natural-language request into a solver configuration."""
    if client is None:
        return {
            "solver_type": "1d_dft",
            "explanation": "未配置 AI 解析服务，已回退到默认一维模型。",
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
                "potential_description": "一维谐振势阱 (AI备份)",
            },
        }

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for item in history:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": f"User Request: {prompt}"})

        completion = client.chat.completions.create(model=model_name, messages=messages, temperature=0.1)
        content = completion.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()

        return json.loads(content)
    except Exception as e:
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
                "potential_description": "一维谐振势阱 (AI备份)",
            },
        }
