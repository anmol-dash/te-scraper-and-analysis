#!/usr/bin/env python3
"""
HPC Client for TE Analysis Pipeline

An interactive client-side application that connects to an HPC cluster,
sets parameters for query.py, previews results, and retrieves output.

Usage:
    python hpc_client.py

Requirements:
    pip install paramiko
"""

import os
import sys
import stat
import getpass
import argparse
import base64
import io
import re
import tarfile
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Error: paramiko is required. Install with: pip install paramiko")
    sys.exit(1)


class HPCClient:
    """Interactive client for running TE analysis on HPC cluster."""

    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.connected = False

        # Default parameters
        self.params = {
            "FAMILY_NAME": "HERVK9",
            "HG38_FA": "/project/amodzlab/index/human/hg38/hg38.fa",
            "BASE_OUT_DIR": "collab_rna",
            "K": 18,
            "TOP_N_GLOBAL": 8,
            "TOP_N_CLUSTER": 5,
            "TOP_N_FORWARD_PRIMERS": 3,
            "MIN_SEQUENCES_FOR_CLUSTERING": 10,
            "all_te_file": "",  # Path to CSV on HPC
            "te_counts": "",    # Path to CSV on HPC
        }

        self.local_output_dir = None
        self.remote_script_path = None
        self.remote_work_dir = None
        self._transport = None
        self._password = None
        self.use_sftp = False

    def connect(self, hostname: str, username: str, password: str, port: int = 22):
        """Connect to the HPC cluster via SSH."""
        print(f"\nConnecting to {hostname}...")

        # Store password for keyboard-interactive auth
        self._password = password
        self._transport = None

        def keyboard_interactive_handler(title, instructions, prompt_list):
            """Handle keyboard-interactive authentication."""
            responses = []
            for prompt, show_input in prompt_list:
                responses.append(self._password)
            return responses

        # Use Transport directly for more control
        try:
            print("Establishing transport...")
            self._transport = paramiko.Transport((hostname, port))
            self._transport.banner_timeout = 60
            self._transport.connect()

            # Try keyboard-interactive auth first (most common for university HPCs)
            print("Authenticating...")
            try:
                self._transport.auth_interactive(username, keyboard_interactive_handler)
            except paramiko.ssh_exception.AuthenticationException:
                # Fall back to password auth
                print("Trying password authentication...")
                self._transport.auth_password(username, password)

            if not self._transport.is_authenticated():
                print("Authentication failed")
                return False

            # Create SSHClient wrapper around the transport
            self.ssh = paramiko.SSHClient()
            self.ssh._transport = self._transport

            # Try to open SFTP (may fail on some HPC systems)
            print("Opening SFTP...")
            try:
                self.sftp = paramiko.SFTPClient.from_transport(self._transport)
                self.use_sftp = True
            except Exception as e:
                print(f"SFTP not available ({e}), will use shell commands for file transfer")
                self.sftp = None
                self.use_sftp = False

            self.connected = True
            print(f"Successfully connected to {hostname}")

            # Get user's home directory
            channel = self._transport.open_session()
            channel.exec_command("echo $HOME")
            self.remote_work_dir = channel.recv(1024).decode().strip()
            channel.close()
            print(f"Remote home directory: {self.remote_work_dir}")

            return True

        except paramiko.ssh_exception.AuthenticationException as e:
            print(f"Authentication failed: {e}")
            return False
        except Exception as e:
            print(f"Connection failed: {e}")
            if self._transport:
                self._transport.close()
            return False

    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self._transport:
            self._transport.close()
        self.connected = False
        print("Disconnected from HPC.")

    def run_command(self, command: str, timeout: int = 300) -> tuple:
        """Execute a command on the remote server."""
        if not self.connected:
            raise RuntimeError("Not connected to HPC")

        channel = self._transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(command)

        # Read output
        out = b""
        err = b""

        while True:
            if channel.recv_ready():
                out += channel.recv(4096)
            if channel.recv_stderr_ready():
                err += channel.recv_stderr(4096)
            if channel.exit_status_ready():
                break

        # Get any remaining data
        while channel.recv_ready():
            out += channel.recv(4096)
        while channel.recv_stderr_ready():
            err += channel.recv_stderr(4096)

        exit_code = channel.recv_exit_status()
        channel.close()

        return out.decode(), err.decode(), exit_code

    def preview_family_count(self) -> int:
        """Preview the number of sequences matching the family name."""
        if not self.params["all_te_file"]:
            print("Error: all_te_file path not set")
            return -1

        family = self.params["FAMILY_NAME"]
        csv_path = self.params["all_te_file"]

        print(f"\nPreviewing sequences for family '{family}'...")

        # First check if file exists
        cmd = f"test -f '{csv_path}' && echo 'exists'"
        out, err, code = self.run_command(cmd)
        if 'exists' not in out:
            print(f"Error: File not found: {csv_path}")
            return -1

        # Count matching rows using grep (case-insensitive)
        # Use grep -c for count, || true to handle no matches (grep returns 1)
        cmd = f"grep -ci '{family}' '{csv_path}' || echo 0"
        out, err, code = self.run_command(cmd)

        try:
            count = int(out.strip().split('\n')[-1])
        except ValueError:
            print(f"Error parsing count: {out}")
            return -1

        print(f"\n{'='*50}")
        print(f"  FILE: {csv_path}")
        print(f"  FAMILY: {family}")
        print(f"  MATCHING SEQUENCES: {count}")
        print(f"{'='*50}")

        return count

    def set_parameter_interactive(self):
        """Interactive menu to set parameters."""
        while True:
            print("\n" + "="*60)
            print("CURRENT PARAMETERS")
            print("="*60)

            param_list = [
                ("1", "FAMILY_NAME", self.params["FAMILY_NAME"]),
                ("2", "HG38_FA", self.params["HG38_FA"]),
                ("3", "BASE_OUT_DIR", self.params["BASE_OUT_DIR"]),
                ("4", "K", self.params["K"]),
                ("5", "TOP_N_GLOBAL", self.params["TOP_N_GLOBAL"]),
                ("6", "TOP_N_CLUSTER", self.params["TOP_N_CLUSTER"]),
                ("7", "TOP_N_FORWARD_PRIMERS", self.params["TOP_N_FORWARD_PRIMERS"]),
                ("8", "MIN_SEQUENCES_FOR_CLUSTERING", self.params["MIN_SEQUENCES_FOR_CLUSTERING"]),
                ("9", "all_te_file", self.params["all_te_file"] or "[NOT SET - REQUIRED]"),
                ("10", "te_counts", self.params["te_counts"] or "[NOT SET - optional]"),
            ]

            # Add descriptions below the parameter list
            print("\n  Parameter descriptions:")
            print("    [9]  all_te_file: TE annotations CSV with columns:")
            print("         chr, start, stop, TE_name, family, strand, + expression columns")
            print("    [10] te_counts: Optional summary CSV (not required for analysis)")

            for num, name, value in param_list:
                print(f"  [{num:>2}] {name:<30} = {value}")

            print("\n  [p]  Preview family count")
            print("  [r]  Run analysis")
            print("  [q]  Quit")
            print("="*60)

            choice = input("\nSelect option (1-10, p, r, q): ").strip().lower()

            if choice == 'q':
                return False
            elif choice == 'p':
                self.preview_family_count()
            elif choice == 'r':
                return True
            elif choice in ['1', '2', '3', '9', '10']:
                # String parameters
                key = param_list[int(choice)-1][1].split()[0]  # Handle "all_te_file (CSV path)"
                current = self.params[key]
                new_val = input(f"Enter new value for {key} [{current}]: ").strip()
                if new_val:
                    self.params[key] = new_val
            elif choice in ['4', '5', '6', '7', '8']:
                # Integer parameters
                key = param_list[int(choice)-1][1]
                current = self.params[key]
                new_val = input(f"Enter new value for {key} [{current}]: ").strip()
                if new_val:
                    try:
                        self.params[key] = int(new_val)
                    except ValueError:
                        print("Invalid integer value")
            else:
                print("Invalid option")

        return False

    def upload_script(self):
        """Upload the analysis script to HPC with current parameters."""
        # Read local query.py
        local_script = Path(__file__).parent / "query.py"

        if not local_script.exists():
            print(f"Error: query.py not found at {local_script}")
            return False

        with open(local_script, 'r') as f:
            script_content = f.read()

        # Create modified script with parameters injected
        param_block = f'''# ==================== CONFIG ====================
FAMILY_NAME = "{self.params['FAMILY_NAME']}"
HG38_FA = "{self.params['HG38_FA']}"
BASE_OUT_DIR = Path("{self.params['BASE_OUT_DIR']}")
K = {self.params['K']}
TOP_N_GLOBAL = {self.params['TOP_N_GLOBAL']}
TOP_N_CLUSTER = {self.params['TOP_N_CLUSTER']}
TOP_N_FORWARD_PRIMERS = {self.params['TOP_N_FORWARD_PRIMERS']}
MIN_SEQUENCES_FOR_CLUSTERING = {self.params['MIN_SEQUENCES_FOR_CLUSTERING']}
'''

        # Add data loading code
        data_load_block = f'''
# ==================== LOAD INPUT DATA ====================
import pandas as pd
print("Loading input data...")
df = pd.read_csv("{self.params['all_te_file']}")
print(f"Loaded all_te_file: {{len(df)}} rows")
'''

        if self.params['te_counts']:
            data_load_block += f'''
df2 = pd.read_csv("{self.params['te_counts']}")
print(f"Loaded te_counts: {{len(df2)}} rows")
'''

        # Find and replace the config section
        # Replace the config block
        config_pattern = r'# ==================== CONFIG ====================.*?# ================================================'
        modified_script = re.sub(config_pattern, param_block + '\n# ================================================',
                                 script_content, flags=re.DOTALL)

        # Insert data loading before "# ==================== LOAD AND FILTER DATA ===================="
        load_marker = "# ==================== LOAD AND FILTER DATA ===================="
        modified_script = modified_script.replace(load_marker, data_load_block + "\n" + load_marker)

        # Upload to remote
        remote_script = f"{self.remote_work_dir}/te_analysis_run.py"

        if self.use_sftp and self.sftp:
            with self.sftp.file(remote_script, 'w') as f:
                f.write(modified_script)
        else:
            # Use shell command with base64 encoding to safely transfer the script
            encoded = base64.b64encode(modified_script.encode()).decode()
            cmd = f"echo '{encoded}' | base64 -d > {remote_script}"
            out, err, code = self.run_command(cmd)
            if code != 0:
                print(f"Failed to upload script: {err}")
                return False

        self.remote_script_path = remote_script
        print(f"Uploaded analysis script to {remote_script}")
        return True

    def run_analysis(self):
        """Run the analysis on HPC."""
        if not self.params["all_te_file"]:
            print("Error: all_te_file path must be set")
            return False

        # Preview count first
        count = self.preview_family_count()
        if count <= 0:
            confirm = input("\nNo sequences found. Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return False

        confirm = input(f"\nProceed with analysis for {count} sequences? (y/n): ").strip().lower()
        if confirm != 'y':
            return False

        # Upload script
        if not self.upload_script():
            return False

        print("\nStarting analysis on HPC...")
        print("This may take a while depending on dataset size...\n")

        # Track installed packages to avoid infinite loops
        installed_packages = set()
        max_retries = 10  # Maximum number of packages to install

        while len(installed_packages) < max_retries:
            # Run the script
            cmd = f"cd {self.remote_work_dir} && python {self.remote_script_path}"
            out, err, code = self.run_command(cmd, timeout=3600)  # 1 hour timeout

            # Check for missing module error
            missing_module = None
            for line in err.split('\n'):
                if 'ModuleNotFoundError: No module named' in line:
                    # Extract module name from error message
                    match = re.search(r"No module named '([^']+)'", line)
                    if match:
                        missing_module = match.group(1).split('.')[0]  # Get base module
                        break

            if missing_module and missing_module not in installed_packages:
                print(f"\nMissing module detected: {missing_module}")
                print(f"Installing {missing_module} via pip...")

                # Map common module names to pip package names
                package_map = {
                    'sklearn': 'scikit-learn',
                    'cv2': 'opencv-python',
                    'PIL': 'Pillow',
                    'bs4': 'beautifulsoup4',
                    'umap': 'umap-learn',
                }
                package_name = package_map.get(missing_module, missing_module)

                # Version constraints for systems with old GCC (avoid source compilation)
                version_constraints = {
                    'scikit-learn': 'scikit-learn<1.4',
                    'numpy': 'numpy<2.0',
                    'scipy': 'scipy<1.12',
                    'pandas': 'pandas<2.2',
                    'umap-learn': 'umap-learn<0.5.5',
                }

                # Try installing with binary-only first (faster, avoids compilation issues)
                install_cmd = f"pip install --user --only-binary :all: {package_name}"
                print(f"Trying: {install_cmd}")
                install_out, install_err, install_code = self.run_command(install_cmd, timeout=300)

                # If binary-only fails, try with version constraints for problematic packages
                if install_code != 0:
                    print("Binary install failed, trying with compatible versions...")
                    constrained_package = version_constraints.get(package_name, package_name)
                    install_cmd = f"pip install --user --only-binary :all: {constrained_package}"
                    print(f"Trying: {install_cmd}")
                    install_out, install_err, install_code = self.run_command(install_cmd, timeout=300)

                # Final fallback: try without binary restriction but with version constraints
                if install_code != 0:
                    print("Still failing, trying source build with version constraints...")
                    constrained_package = version_constraints.get(package_name, package_name)
                    install_cmd = f"pip install --user {constrained_package}"
                    print(f"Trying: {install_cmd}")
                    install_out, install_err, install_code = self.run_command(install_cmd, timeout=600)

                if install_code == 0:
                    print(f"Successfully installed {package_name}")
                    installed_packages.add(missing_module)
                    print("Retrying analysis...\n")
                    continue
                else:
                    print(f"Failed to install {package_name}:")
                    print(install_err)
                    break
            else:
                # No missing module or already tried installing it
                break

        print("="*60)
        print("ANALYSIS OUTPUT")
        print("="*60)
        print(out)

        if err:
            print("\nSTDERR:")
            print(err)

        if code == 0:
            print("\nAnalysis completed successfully!")
            if installed_packages:
                print(f"Packages installed during run: {', '.join(installed_packages)}")
            return True
        else:
            print(f"\nAnalysis failed with exit code {code}")
            return False

    def generate_clustering_plots(self):
        """Generate clustering plots after analysis."""
        if not self.connected:
            print("Not connected to HPC")
            return False

        family = self.params["FAMILY_NAME"].lower()
        base_dir = self.params['BASE_OUT_DIR']
        
        # Using string formatting with double curly braces
        remote_script = """
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram.
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

# Set up paths
base_dir = Path('{{base_dir}}')
family = '{{family}}'
output_dir = base_dir / family
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
try:
    # Try to load the preprocessed data
    data_path = output_dir / "{{family}}_clustering_data.csv"
    if not data_path.exists():
        print("Error: Clustering data not found at {{}}".format(str(data_path)))
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    print("Loaded clustering data with {{}} sequences".format(len(df)))
    
    # Prepare data for clustering
    sequences = df['sequence'].values
    X = df.drop(columns=['sequence']).values
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    n_clusters = min(10, len(sequences))  # Cap at 10 clusters
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',
        linkage='ward',
        compute_distances=True
    )
    clusters = model.fit_predict(X)
    
    # Plot dendrogram
    print("Generating dendrogram...")
    plt.figure(figsize=(12, 8))
    plot_dendrogram(model, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram - {{family}}'.format(family=family))
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / "{{family}}_dendrogram.png".format(family=family)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Saved dendrogram to {{}}".format(str(plot_path)))
    
    # Generate consensus sequences
    print("Generating consensus sequences...")
    df['cluster'] = clusters
    
    # For each cluster, generate consensus
    for cluster_id in range(n_clusters):
        cluster_seqs = df[df['cluster'] == cluster_id]['sequence'].tolist()
        if not cluster_seqs:
            continue
            
        # Generate consensus (simple majority voting)
        consensus = []
        max_len = max(len(seq) for seq in cluster_seqs)
        
        for i in range(max_len):
            # Count bases at position i
            bases = {{}}
            for seq in cluster_seqs:
                if i < len(seq):
                    base = seq[i].upper()
                    bases[base] = bases.get(base, 0) + 1
            
            # Get most common base (or gap if no clear consensus)
            if bases:
                consensus_base = max(bases.items(), key=lambda x: x[1])[0]
                consensus.append(consensus_base)
            else:
                consensus.append('-')
        
        # Save consensus sequence (with gaps)
        consensus_seq = ''.join(consensus)
        consensus_path = output_dir / ("{{}}_cluster_{{}}_consensus.txt".format(family, cluster_id+1))
        with open(consensus_path, 'w') as f:
            f.write(">consensus_cluster_{{}}\n{{}}\n".format(cluster_id+1, consensus_seq))
        
        # Generate gap-reduced consensus
        gap_reduced = [base for base in consensus if base != '-']
        gap_reduced_seq = ''.join(gap_reduced)
        gap_reduced_path = output_dir / ("{{}}_cluster_{{}}_consensus_gap_reduced.txt".format(family, cluster_id+1))
        with open(gap_reduced_path, 'w') as f:
            f.write(">consensus_cluster_{{}}_gap_reduced\n{{}}\n".format(cluster_id+1, gap_reduced_seq))
    
    print("Consensus sequences generated successfully!")
    
except Exception as e:
    print("Error generating clustering plots: {{}}".format(str(e)), file=sys.stderr)
    sys.exit(1)
""".format(
            base_dir=self.params['BASE_OUT_DIR'],
            family=family
        )

        # Format the script with the actual values
        formatted_script = remote_script.replace('{{base_dir}}', base_dir.replace("'", "\\'")) \
                                       .replace('{{family}}', family.replace("'", "\\'"))

        # Save script to a temporary file
        script_path = "/tmp/generate_plots_{}.py".format(os.getpid())

        if self.use_sftp and self.sftp:
            with self.sftp.file(script_path, 'w') as f:
                f.write(formatted_script)
        else:
            # Use shell command with base64 encoding to safely transfer the script
            encoded = base64.b64encode(formatted_script.encode()).decode()
            cmd = f"echo '{encoded}' | base64 -d > {script_path}"
            out, err, code = self.run_command(cmd)
            if code != 0:
                print(f"Failed to upload clustering script: {err}")
                return False

        # Make script executable
        self.run_command("chmod +x {}".format(script_path))

        # Run the script
        print("Generating clustering plots and consensus sequences...")
        out, err, code = self.run_command("python {}".format(script_path), timeout=600)

        # Clean up
        self.run_command("rm -f {}".format(script_path))

        if code != 0:
            print("Error generating plots: {}".format(err))
            return False

        print(out)
        return True

    def retrieve_results(self, local_dir: str):
        """Download results from HPC to local directory."""
        # Generate clustering plots and consensus sequences first
        print("\nGenerating clustering plots and consensus sequences...")
        if not self.generate_clustering_plots():
            print("Warning: Failed to generate clustering plots. Continuing with available results...")

        # Expand ~ and make path absolute
        local_path = Path(local_dir).expanduser()
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path

        try:
            local_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"Error: Permission denied creating directory: {local_path}")
            print("Please check the path and try again.")
            return
        except OSError as e:
            print(f"Error creating directory: {e}")
            print("Hint: Use '~' for home directory (e.g., ~/Documents/output)")
            print("      Or use relative path (e.g., ./output)")
            return

        # Remote output directory
        family = self.params["FAMILY_NAME"].lower()
        remote_out = f"{self.remote_work_dir}/{self.params['BASE_OUT_DIR']}/{family}"

        print(f"\nRetrieving results from {remote_out}")
        print(f"Saving to {local_path}")

        if self.use_sftp and self.sftp:
            def download_recursive(remote_path, local_base):
                """Recursively download directory."""
                try:
                    items = self.sftp.listdir_attr(remote_path)
                except IOError:
                    print(f"Cannot access {remote_path}")
                    return

                for item in items:
                    remote_item = f"{remote_path}/{item.filename}"
                    local_item = local_base / item.filename

                    if stat.S_ISDIR(item.st_mode):
                        local_item.mkdir(exist_ok=True)
                        download_recursive(remote_item, local_item)
                    else:
                        print(f"  Downloading: {item.filename}")
                        self.sftp.get(remote_item, str(local_item))

            download_recursive(remote_out, local_path)
        else:
            # Use tar + direct binary transfer through SSH channel
            print("Packaging results (this may take a moment)...")

            # Check if directory exists
            out, err, code = self.run_command(f"test -d '{remote_out}' && echo 'exists'")
            if 'exists' not in out:
                print(f"Error: Remote directory not found: {remote_out}")
                return

            # Create tar archive on remote
            remote_tar = f"/tmp/te_results_{os.getpid()}.tar.gz"
            cmd = f"cd '{remote_out}' && tar -czf {remote_tar} ."
            print("Creating archive on remote...")
            out, err, code = self.run_command(cmd, timeout=600)

            if code != 0:
                print(f"Error creating archive: {err}")
                return

            # Get file size
            out, err, code = self.run_command(f"stat -c%s {remote_tar} 2>/dev/null || stat -f%z {remote_tar}")
            try:
                file_size = int(out.strip())
                print(f"Archive size: {file_size / 1024 / 1024:.2f} MB")
            except:
                file_size = 0

            # Transfer binary file directly through SSH channel
            print("Transferring archive...")
            local_tar = local_path / "results.tar.gz"

            try:
                channel = self._transport.open_session()
                channel.exec_command(f"cat {remote_tar}")

                with open(local_tar, 'wb') as f:
                    bytes_received = 0
                    while True:
                        data = channel.recv(65536)  # 64KB buffer
                        if not data:
                            break
                        f.write(data)
                        bytes_received += len(data)
                        if file_size > 0:
                            pct = min(100, bytes_received * 100 / file_size)
                            print(f"  Progress: {pct:.1f}% ({bytes_received / 1024 / 1024:.1f} MB)", end='\r')

                channel.close()
                print(f"\n  Received {bytes_received / 1024 / 1024:.2f} MB")

            except Exception as e:
                print(f"\nError during transfer: {e}")
                return

            # Clean up remote tar
            self.run_command(f"rm -f {remote_tar}")

            print("Extracting files...")

            # Extract locally
            try:
                tar_file = tarfile.open(local_tar, mode='r:gz')
                tar_file.extractall(local_path)
                tar_file.close()
                os.remove(local_tar)  # Clean up local tar
            except Exception as e:
                print(f"Error extracting: {e}")
                print(f"Archive saved at: {local_tar}")
                return

        print(f"\nResults saved to {local_path}")
        self.local_output_dir = local_path

    def main_menu(self):
        """Main interactive menu after connection."""
        while True:
            try:
                print("\n" + "="*60)
                print("HPC TE ANALYSIS CLIENT")
                print("="*60)
                print("  [1] Configure parameters")
                print("  [2] Preview family count")
                print("  [3] Run full analysis")
                print("  [4] Retrieve results")
                print("  [5] Disconnect and exit")
                print("="*60)

                choice = input("\nSelect option (1-5): ").strip()

                if choice == '1':
                    if self.set_parameter_interactive():
                        # User chose to run
                        self.run_analysis()
                elif choice == '2':
                    self.preview_family_count()
                elif choice == '3':
                    self.run_analysis()
                elif choice == '4':
                    if self.local_output_dir:
                        default_dir = str(self.local_output_dir)
                        local_dir = input(f"Enter local output directory [{default_dir}]: ").strip()
                        local_dir = local_dir or default_dir
                    else:
                        local_dir = input("Enter local output directory (e.g., ~/Documents/output): ").strip()
                    if local_dir:
                        self.retrieve_results(local_dir)
                elif choice == '5':
                    break
                else:
                    print("Invalid option")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                confirm = input("Exit? (y/n): ").strip().lower()
                if confirm == 'y':
                    break
            except Exception as e:
                print(f"\nError: {e}")
                print("Returning to main menu...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive HPC client for TE analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hpc_client.py
  python hpc_client.py --host cluster.university.edu --user myusername
  python hpc_client.py -H cluster.edu -u myuser -o ./results
        """
    )
    parser.add_argument("-H", "--host", help="HPC hostname")
    parser.add_argument("-p", "--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("-u", "--user", help="Username")
    parser.add_argument("-o", "--output", help="Local output directory for results")
    args = parser.parse_args()

    print("="*60)
    print("  HPC TE ANALYSIS CLIENT")
    print("  Interactive client for running TE analysis on HPC")
    print("="*60)

    client = HPCClient()

    # Get connection details (use args or prompt)
    print("\nEnter HPC connection details:")

    hostname = args.host
    if not hostname:
        hostname = input("  Hostname: ").strip()
    else:
        print(f"  Hostname: {hostname}")

    if not hostname:
        print("Hostname is required")
        sys.exit(1)

    port = args.port
    if port == 22:
        port_str = input("  Port [22]: ").strip()
        port = int(port_str) if port_str else 22

    username = args.user
    if not username:
        username = input("  Username: ").strip()
    else:
        print(f"  Username: {username}")

    if not username:
        print("Username is required")
        sys.exit(1)

    password = getpass.getpass("  Password: ")

    # Connect
    if not client.connect(hostname, username, password, port):
        print("Failed to connect. Exiting.")
        sys.exit(1)

    # Store output directory if provided
    if args.output:
        client.local_output_dir = Path(args.output)
        print(f"Results will be saved to: {args.output}")

    try:
        # Show main menu
        client.main_menu()
    finally:
        client.disconnect()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
