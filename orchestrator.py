import os
import time
import subprocess
import logging
import sys
import psutil  # pip install psutil

# --- CONFIGURATION ---
FOLDERS = {
    "ideas": "0_ideas",
    "specs": "1_specs",
    "wip": "2_in_progress",
    "done": "3_done"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

workers = {}


def count_files(folder):
    if not os.path.exists(folder): os.makedirs(folder)
    return len([f for f in os.listdir(folder) if not f.startswith('.')])


def is_running(role):
    """Checks if the agent process is alive using psutil"""
    if role not in workers: return False

    proc = workers[role]

    # 1. Check Python handle
    if proc.poll() is None: return True

    # 2. Deep check for Windows (shell=True creates wrapper processes)
    try:
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = " ".join(p.info['cmdline'] or [])
                # We check for the specific agent role in the command line
                if 'opencode' in cmd and role in cmd:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    # If we get here, it's really dead.
    if role in workers: del workers[role]
    return False


def run_agent(role, instruction):
    if is_running(role): return

    logging.info(f"🟢 STARTING: {role.upper()}")

    # Open log file directly
    log_file = open(f"logs_{role}.txt", "a", encoding="utf-8")

    try:
        # We call the executable directly.
        # On Windows, we use 'opencode.cmd' or 'opencode.exe'
        # because 'opencode' is usually a shim.
        executable = "opencode.cmd" if os.name == 'nt' else "opencode"

        cmd = [executable, "run", instruction, "--agent", role]

        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # Ensure it never waits for input
            env=os.environ.copy(),
            shell=False  # Removing shell avoids the CMD wrapper hang
        )
        workers[role] = p

    except FileNotFoundError:
        logging.error("❌ 'opencode' command not found. Try using shell=True or full path.")
    except Exception as e:
        logging.error(f"❌ Launch failed: {e}")


def main():
    logging.info("🏭 FACTORY ONLINE (Native Shell Mode).")
    for f in FOLDERS.values(): os.makedirs(f, exist_ok=True)

    while True:
        try:
            # 1. READ STATE
            n_ideas = count_files(FOLDERS["ideas"])
            n_specs = count_files(FOLDERS["specs"])
            n_wip = count_files(FOLDERS["wip"])

            any_active = is_running("designer") or is_running("architect") or is_running("coder")

            # 2. RELAY RACE LOGIC

            # STAGE 3: CODING
            if n_specs > 0:
                if not any_active:
                    run_agent("coder", "Process the pending spec in 1_specs/.")
                else:
                    sys.stdout.write(f"\r⏳ Pipeline working... (Specs: {n_specs})   ")

            # STAGE 2: ARCHITECTURE
            elif n_ideas > 0:
                if not any_active:
                    run_agent("architect", "Process the pending idea in 0_ideas/.")
                else:
                    sys.stdout.write(f"\r⏳ Pipeline working... (Ideas: {n_ideas})   ")

            # STAGE 1: DESIGN
            elif n_ideas == 0 and n_specs == 0 and n_wip == 0:
                if not any_active:
                    logging.info("\n✨ Pipeline empty. Triggering Designer.")
                    run_agent("designer", "Generate ONE new feature idea in 0_ideas/.")
                else:
                    sys.stdout.write(f"\r⏳ Designer is thinking...   ")

            # WIP Watchdog
            elif n_wip > 0 and not any_active:
                logging.warning("\n⚠️ WIP detected but Coder is sleeping. Waking Coder.")
                run_agent("coder", "Resume work on 2_in_progress/.")

            sys.stdout.flush()
            time.sleep(5)

        except KeyboardInterrupt:
            logging.info("\n🛑 STOPPING.")
            # On Windows shell=True, we often need to kill the process tree manually
            # This loops through workers and kills them
            for p in workers.values():
                try:
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
                except:
                    pass
            break


if __name__ == "__main__":
    main()