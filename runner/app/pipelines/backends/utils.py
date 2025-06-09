
import subprocess
import re
PIPELINE_CONF = "/etc/supervisor/conf.d/{pipeline_id}.conf"
init_pyenv = 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)"'

def set_backend_device_and_port(pipeline_id, cuda_device=0):
    pipeline_conf_file = PIPELINE_CONF.format(pipeline_id=pipeline_id)
    with open(pipeline_conf_file, 'r') as f:
        lines = f.readlines()

    device_pattern = re.compile(r'(--cuda-device )(\S+)')
    updated_lines = []

    for line in lines:
        if line.strip().startswith('command=') and '--cuda-device ' in line:
            line = device_pattern.sub(f"--cuda-device {cuda_device}", line)
        
        print(f"Updated --cuda-device and port: {line}")
        updated_lines.append(line)

    with open(pipeline_conf_file, 'w') as f:
        f.writelines(updated_lines)

    # update supervisor configuration
    subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock reread", shell=True, check=True)
    subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock update", shell=True, check=True)

def run_command(command):
    """
    Run a shell command and return the output.
    """
    command = init_pyenv+' '+command
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command '{command}' failed with error: {e.stderr}")

def start_backend(pipeline_id, cuda_device):
    """
    Start the backend using the provided command.
    """
    if pipeline_id != "comfyui-playground":
        set_backend_device_and_port(pipeline_id, cuda_device)

    try:
        result = subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock start {pipeline_id}", shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to start backend {pipeline_id}: {e.stderr}")

def stop_backend(pipeline_id):
    """
    Stop the backend using the provided command.
    """
    try:
        result = subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock stop {pipeline_id}", shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to stop backend {pipeline_id}: {e.stderr}")
    
def create_pipeline_runner_config(pipeline_id, port, venv_name):

    """
    Create a pipeline runner config file.
    """
    config = """
    [program:{pipeline_id}]
    command=/bin/bash -c 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate $PIPELINE_VENV && python -u /app/workspace/main.py --disable-smart-memory --disable-cuda-malloc --listen 0.0.0.0 --port {port} --cuda-device 0'
    autostart=false
    startretries=0
    priority=3
    stdout_logfile=/dev/fd/1
    stdout_logfile_maxbytes=0
    redirect_stderr=true
    autorestart=true
    environment=PYTHONUNBUFFERED=1,PIPELINE_VENV={venv_name},PYTHONPATH=/root/.pyenv/versions/comfyui-base
    """
    config = config.format(pipeline_id=pipeline_id, port=port, venv_name=venv_name)
    with open(f"/etc/supervisor/conf.d/{pipeline_id}.conf", "w") as f:
        f.write(config)
    
    subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock reread", shell=True, check=True)
    subprocess.run(f"supervisorctl -s unix:///tmp/supervisor.sock update", shell=True, check=True)