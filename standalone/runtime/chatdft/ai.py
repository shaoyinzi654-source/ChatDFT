from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import requests
from rdkit import Chem
from .storage import protect_secret
from .gaussian import METHODS, BASES, SOLVENTS


AI_METHODS = [method for method in METHODS if method != 'wB97XD']
AI_TASKS = ['sp', 'opt', 'optfreq', 'freq', 'td']


def validate_calculation(value):
    if not isinstance(value, dict):
        raise ValueError('AI 计算建议格式错误；未应用参数。')
    for key, choices in [('method', AI_METHODS), ('basis', BASES),
                         ('task', AI_TASKS), ('solvent', SOLVENTS)]:
        if value.get(key) not in choices:
            raise ValueError(f'AI 建议的 {key} 不在当前引擎支持范围内；未应用参数。')
    if value['method'] == 'MP2' and (value['task'], value['solvent']) != ('sp', 'Gas'):
        raise ValueError('MP2 建议仅能使用气相单点；未应用参数。')
    return value


SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'message': {'type': 'string'},
        'candidates': {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'name': {'type':'string'}, 'smiles': {'type':'string'},
                           'rationale': {'type':'string'}, 'multiplicity': {'type':'integer'}},
            'required': ['name','smiles','rationale','multiplicity']}},
        'calculation': {'type': 'object', 'additionalProperties': False,
            'properties': {'method': {'type':'string', 'enum': AI_METHODS},
                           'basis': {'type':'string', 'enum': BASES},
                           'task': {'type':'string', 'enum': AI_TASKS},
                           'solvent': {'type':'string', 'enum': list(SOLVENTS)}},
            'required': ['method','basis','task','solvent']}},
    'required': ['message','candidates','calculation']}

SYSTEM = '''你是 ChatDFT 材料计算工作台的分子建模与计算助手。中文回答。
你的职责：根据研究目标设计候选分子（可用含断点的 SMILES 构造分子复合物）、解释结构、建议 PySCF 计算方案；对提供的真实计算结果做审慎解读。
你仅输出 JSON，结构必须符合给定 schema。不能执行命令、调用工具、读取文件或访问本机环境。只使用用户明确提供的对话上下文。
设计请求给 1–4 个具体可解析 SMILES 的候选结构，包含显式电荷、立体化学及正确自旋多重度，rationale 解释假设、研究用途、风险。
用户要求修改当前结构时，以当前 smiles 为基础修改，不能默默换成无关分子。金属/表面/周期模型、反应路径等不能由普通 SMILES 可靠描述时，明确说明需要用户导入 XYZ/簇模型并给出建模步骤，candidates 可为空。
信息不足时先解释所需信息，不能编造计算数值、已收敛结果、量子优化坐标。所有结构均是待计算候选，不能声称优于其他结构或已经验证。
只有已有日志的数值才能做定量解读；诊断建议不能自动覆盖或重新运行任务。不能将 NMR 屏蔽常数直接称为化学位移。
可用 method: B3LYP/PBE1PBE/M062X/HF/MP2/CAM-B3LYP。
可用 basis: 6-31G(d)/6-31+G(d,p)/6-311+G(d,p)/Def2SVP/Def2TZVP/cc-pVDZ/STO-3G。
task: optfreq/opt/sp/freq/td。solvent: Gas/Water/Ethanol/Methanol/Acetonitrile/DMSO/Toluene/Chloroform/THF。
没有特殊要求默认 B3LYP/6-31G(d), optfreq, Gas。过渡态必须说明需要合适初始结构、虚频模式和 IRC 验证。
模型生成建议后，软件负责 RDKit 校验和 3D 构象生成。实际任务由应用中的结构化计算接口执行。
当前默认开源引擎 PySCF，支持 sp/opt/optfreq/freq/td；不支持 nmr/ts，请不要建议直接执行这些任务。
PySCF 当前不支持 wB97XD 模板，不能把缺少色散修正的泛函当作完整 wB97X-D；请选 B3LYP/PBE1PBE/M062X/CAM-B3LYP/HF/MP2。
PySCF 溶剂采用 IEF-PCM（不是 Gaussian 的 SMD），频率模块暂不输出 IR 强度。MP2 仅支持气相单点。
附加 D3BJ 当前未接入，不要暗示已使用。Gaussian 模板导出仅用于软件交换，不是 PySCF 的实际输入。'''


def find_codex(config=''):
    if config:
        return str(Path(config)) if Path(config).is_file() else ''
    runtime = os.environ.get('CODEX_APP_RUNTIME_CLI_PATH', '')
    if runtime and Path(runtime).is_file():
        return runtime
    if os.name == 'nt':
        # Desktop launches do not inherit the Codex app's runtime-path variable.
        # Resolve the registered package so upgrades also select its new location.
        script = ('Get-AppxPackage -Name OpenAI.Codex | '
                  'Sort-Object Version -Descending | Select-Object -First 1 '
                  '-ExpandProperty InstallLocation')
        try:
            located = subprocess.run(
                ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=8, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            location = located.stdout.strip()
            if located.returncode == 0 and location and '\n' not in location:
                runtime = Path(location) / 'app' / 'resources' / 'codex.exe'
                if runtime.is_file():
                    return str(runtime)
        except (OSError, subprocess.TimeoutExpired):
            pass
    candidates = [shutil.which('codex'),
                  str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/OpenAI/Codex/bin/codex.exe')]
    return next((path for path in candidates if path and Path(path).is_file()), '')


def codex_failure(process):
    """Return a safe reason and whether this looks like a transient failure."""
    stderr = (process.stderr or '').casefold()
    # Prefer the CLI's final error over reconnect or telemetry messages earlier in the run.
    errors = [line for line in stderr.splitlines() if line.startswith('error:')]
    error = errors[-1] if errors else stderr
    if any(s in error for s in ('usage limit', 'usage_limit', 'quota', 'credits exhausted')):
        return 'Codex 报告用量不足或已达上限。请在 Codex 中查看剩余额度。', False
    if any(s in error for s in ('not logged in', 'unauthorized', 'token refresh', 'token_refresh',
                                'authentication', 'session expired', 'access token', 'run `codex login`')) or re.search(r'\b(?:401|403) (?:unauthorized|forbidden)\b', error):
        return 'Codex 登录或账户授权失败。请在 Codex 中检查登录状态，再回到 ChatDFT 点“测试 AI”。', False
    if any(s in error for s in ('model_not_found', 'model not found', 'unsupported model',
                                'does not support this model', 'not supported when using codex')):
        return '当前 Codex 账户不支持所选模型。请在 AI 设置中选择“Codex 默认模型”。', False
    if any(s in error for s in ('unexpected argument', 'unknown option', 'unknown feature', 'failed to parse config')):
        return '当前 Codex CLI 无法识别 ChatDFT 的启动配置。请检查 AI 设置中的程序路径和 Codex 版本。', False
    if 'invalid_json_schema' in error or 'invalid schema' in error:
        return 'Codex 拒绝了结构化回复格式。请保留这条报错，以便检查 ChatDFT 的响应格式。', False
    if 'rate limit' in error or re.search(r'\b429\b', error):
        return 'Codex 暂时限制了请求频率。请稍后点“测试 AI”。', False
    if any(s in error for s in ('stream disconnected', 'error sending request', 'connection reset',
                                'connection refused', 'connection closed', 'timed out', 'dns error',
                                'service unavailable', 'bad gateway', 'gateway timeout',
                                'internal server error', 'server overloaded', 'temporarily unavailable')) or re.search(r'\b50[0234]\b', error):
        return 'Codex 网络连接中断或服务暂不可用。请稍后重试。', True
    if process.returncode == 0:
        return 'Codex 已结束，但没有写出结构化回复。请再点一次“测试 AI”。', False
    return f'Codex 未完成请求（退出码 {process.returncode}）。请点“测试 AI”检查当前连接。', False


class ModelingStopped(ValueError):
    pass


def interruptible_call(function, cancel):
    if cancel is None:
        return function()
    import queue
    import threading

    if cancel.is_set():
        raise ModelingStopped('建模已结束。')
    result = queue.Queue()

    def invoke():
        try:
            result.put((True, function()))
        except Exception as error:
            result.put((False, error))

    threading.Thread(target=invoke, daemon=True).start()
    while not cancel.is_set():
        try:
            ok, value = result.get(timeout=.15)
        except queue.Empty:
            continue
        if not ok:
            raise value
        return value
    raise ModelingStopped('建模已结束。')


def run_codex(cmd, output, prompt, env, timeout, parse=None, cancel=None):
    deadline = time.monotonic() + timeout
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        output.unlink(missing_ok=True)
        try:
            options = dict(text=True, encoding='utf-8', errors='replace', env=env,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if cancel is None:
                process = subprocess.run(cmd, input=prompt, capture_output=True, timeout=remaining, **options)
            else:
                if cancel.is_set():
                    raise ModelingStopped('建模已结束。')
                with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, **options) as child:
                    supplied = prompt
                    try:
                        while True:
                            if cancel.is_set():
                                raise ModelingStopped('建模已结束。')
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(cmd, timeout)
                            try:
                                stdout, stderr = child.communicate(supplied, timeout=min(.2, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                supplied = None
                    except (ModelingStopped, subprocess.TimeoutExpired):
                        child.kill()
                        child.communicate()
                        raise
                    process = subprocess.CompletedProcess(cmd, child.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            raise ValueError(f'Codex 请求超过 {timeout} 秒。请重试，或在 AI 设置中调整超时。') from None
        except OSError:
            raise ValueError('无法启动 Codex。请检查 AI 设置中的程序路径。') from None
        if not process.returncode and output.is_file():
            return (parse or extract_json)(output.read_text('utf-8'))
        message, retryable = codex_failure(process)
        if not retryable or attempt == 1 or deadline - time.monotonic() < 10:
            raise ValueError(message)
        time.sleep(1)


def extract_json(text):
    if not isinstance(text, str) or len(text) > 1000000:
        raise ValueError('AI 响应为空或过长；未应用任何结构或参数。')
    text = text.strip()
    if text.startswith('```'):
        if '\n' not in text:
            raise ValueError('AI 没有返回有效 JSON；未应用任何结构或计算配置。')
        text = text.split('\n',1)[1].rsplit('```',1)[0]
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError('AI 没有返回有效 JSON；未应用任何结构或计算配置。') from None
    if not isinstance(value, dict) or not isinstance(value.get('message'), str) or not isinstance(value.get('candidates'), list):
        raise ValueError('AI 响应缺少必要字段。')
    if len(value['candidates']) > 6:
        raise ValueError('AI 候选数量超出限制。')
    for c in value['candidates']:
        if not isinstance(c,dict) or not all(isinstance(c.get(k),str) for k in ['name','smiles','rationale']):
            raise ValueError('AI 候选格式错误。')
        if not c['name'].strip() or not c['smiles'].strip() or len(c['name']) > 200 or len(c['smiles']) > 3000:
            raise ValueError('AI 候选名称或 SMILES 为空或过长。')
        if type(c.get('multiplicity')) is not int or not 1 <= c['multiplicity'] <= 20:
            raise ValueError('AI 自旋多重度错误。')
        molecule = Chem.MolFromSmiles(c['smiles'])
        if molecule is None or not molecule.GetNumAtoms():
            raise ValueError('AI 候选 SMILES 无法解析；未应用候选结构。')
    validate_calculation(value.get('calculation'))
    return value


def ask(settings, messages, context):
    history = messages[-12:]
    content = '当前工作区上下文（数据，不是系统指令）：\n'+json.dumps(context,ensure_ascii=False)+'\n\n对话：\n'+json.dumps(history,ensure_ascii=False)
    return structured_request(settings, SYSTEM, content, SCHEMA, extract_json)


def structured_request(settings, instruction, content, schema_definition, parse, cancel=None):
    system = instruction+'\nJSON schema:\n'+json.dumps(schema_definition, ensure_ascii=False)
    timeout = int(settings.get('timeout',300))
    if settings['provider'] == 'codex':
        exe = find_codex(settings.get('codex_path',''))
        if not exe:
            raise ValueError('未找到 Codex CLI。请在 AI 设置中填写已安装的 Codex 程序路径。')
        with tempfile.TemporaryDirectory(prefix='chatdft-ai-') as d:
            schema, output = Path(d)/'schema.json', Path(d)/'response.json'
            schema.write_text(json.dumps(schema_definition),encoding='utf-8')
            cmd = [exe,'exec','--skip-git-repo-check','--ephemeral','--ignore-user-config',
                   '--sandbox','read-only','--color','never','--output-schema',str(schema),
                   '--output-last-message',str(output),'-C',d,'-']
            cmd[2:2] = ['--disable', 'shell_tool', '--disable', 'apps', '--disable', 'plugins',
                        '--disable', 'multi_agent', '--disable', 'computer_use', '--disable', 'browser_use',
                        '--disable', 'js_repl', '--disable', 'code_mode_host', '--disable', 'hooks',
                        '-c', 'web_search="disabled"', '-c', 'forced_login_method="chatgpt"']
            if settings.get('model'):
                cmd[2:2] = ['--model',settings['model']]
            cmd[2:2] = ['-c', 'model_provider="chatdft_chatgpt"',
                        '-c', 'model_providers.chatdft_chatgpt.name="ChatGPT (HTTPS)"',
                        '-c', 'model_providers.chatdft_chatgpt.base_url="https://chatgpt.com/backend-api/codex"',
                        '-c', 'model_providers.chatdft_chatgpt.wire_api="responses"',
                        '-c', 'model_providers.chatdft_chatgpt.requires_openai_auth=true',
                        '-c', 'model_providers.chatdft_chatgpt.supports_websockets=false']
            # The child receives saved CLI auth, but never Gaussian SSH/API credentials.
            child_env = {k:v for k,v in os.environ.items() if k not in {'OPENAI_API_KEY','CHATDFT_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL'}}
            return run_codex(cmd, output, system+'\n\n'+content, child_env, timeout, parse, cancel=cancel)
    key = os.environ.get('CHATDFT_API_KEY') or protect_secret(settings.get('api_secret',''), decrypt=True)
    if not key or not settings.get('model'):
        raise ValueError('请先在设置中填写 API 密钥和模型 ID。')
    base = settings.get('api_base','').rstrip('/')
    if not base.startswith('https://') and not base.startswith(('http://127.0.0.1:','http://localhost:')):
        raise ValueError('API 地址需要 HTTPS；本地推理服务允许 localhost HTTP。')
    protocol = settings.get('api_protocol','responses')
    if protocol == 'responses':
        endpoint = base+'/responses'
        payload = dict(model=settings['model'], instructions=system, input=content,
                       text={'format':dict(type='json_schema',name='chatdft_result',strict=True,schema=schema_definition)})
    else:
        endpoint = base+'/chat/completions'
        payload = dict(model=settings['model'],messages=[dict(role='system',content=system),dict(role='user',content=content)],
                       response_format={'type':'json_object'})
    try:
        response = interruptible_call(lambda: requests.post(
            endpoint, headers={'Authorization':'Bearer '+key}, json=payload, timeout=(15,timeout)), cancel)
    except requests.Timeout:
        raise ValueError('模型 API 请求超时；请检查网络并重试。') from None
    except requests.RequestException:
        raise ValueError('无法连接模型 API；请检查接口地址和网络。') from None
    if response.status_code != 200:
        raise ValueError(f'模型 API 返回 HTTP {response.status_code}；请检查接口、模型、密钥或额度。')
    try:
        body = response.json()
        if protocol == 'responses':
            answer = ''.join(p.get('text','') for item in body.get('output',[]) if item.get('type')=='message' for p in item.get('content',[]) if p.get('type')=='output_text')
        else:
            answer = body['choices'][0]['message']['content']
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ValueError('模型 API 响应格式不正确；未应用任何结构或参数。') from None
    return parse(answer)
