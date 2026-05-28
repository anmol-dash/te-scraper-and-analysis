import { open } from "@tauri-apps/plugin-dialog";
import { useCallback, useEffect, useState } from "react";
import { hpcBatchParallel, pyCancel, pyRun, pyRunFull } from "@/lib/ipc";
import { useAppStore } from "@/store/appStore";

// ── Local pipeline panel ────────────────────────────────────────────────────

function LocalPanel() {
  const [family, setFamily] = useState("");
  const [assembly, setAssembly] = useState("hg38");
  const [genome, setGenome] = useState("");
  const [maxLoci, setMaxLoci] = useState("");
  const [exprCsv, setExprCsv] = useState("");
  const [outDir, setOutDir] = useState("results");

  const runPhase = useAppStore((s) => s.runPhase);
  const runId = useAppStore((s) => s.runId);
  const beginRun = useAppStore((s) => s.beginRun);
  const finishRun = useAppStore((s) => s.finishRun);
  const resetRun = useAppStore((s) => s.resetRun);
  const pushToast = useAppStore((s) => s.pushToast);
  const busy = runPhase === "running";

  const pickFile = useCallback(
    async (setter: (v: string) => void, filters?: { name: string; extensions: string[] }[]) => {
      try {
        const path = await open({ multiple: false, directory: false, filters });
        if (typeof path === "string") setter(path);
      } catch (e) {
        pushToast(e instanceof Error ? e.message : String(e));
      }
    },
    [pushToast],
  );

  const run = useCallback(async () => {
    if (!family.trim()) {
      pushToast("TE family name is required.");
      return;
    }
    const argv: string[] = [
      "query.py",
      "--local",
      "--family", family.trim(),
      "--assembly", assembly,
      "--output", outDir.trim() || "results",
    ];
    if (genome.trim()) argv.push("--genome", genome.trim());
    else argv.push("--skip-genome");
    if (maxLoci.trim()) argv.push("--max-loci", maxLoci.trim());
    if (exprCsv.trim()) argv.push("--expression-assembly", exprCsv.trim());
    try {
      beginRun(globalThis.crypto.randomUUID());
      await pyRun(argv);
      finishRun("done");
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
      finishRun("errored");
    }
  }, [family, assembly, genome, maxLoci, exprCsv, outDir, beginRun, finishRun, pushToast]);

  const cancel = useCallback(async () => {
    if (!runId) return;
    try {
      await pyCancel(runId);
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
    } finally {
      resetRun();
    }
  }, [runId, pushToast, resetRun]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-3">

        <Field label="TE Family *" hint="e.g. THE1D-int, HERVK">
          <input
            className={input}
            value={family}
            onChange={(e) => setFamily(e.target.value)}
            placeholder="THE1D-int"
            disabled={busy}
          />
        </Field>

        <Field label="Assembly">
          <select className={input} value={assembly} onChange={(e) => setAssembly(e.target.value)} disabled={busy}>
            <option value="hg38">hg38 (human)</option>
            <option value="hg19">hg19 (human)</option>
            <option value="mm10">mm10 (mouse)</option>
            <option value="mm39">mm39 (mouse)</option>
          </select>
        </Field>

        <Field label="Genome FASTA" hint="Local path on this machine (optional)">
          <div className="flex gap-2">
            <input className={`${input} min-w-0 flex-1`} value={genome} onChange={(e) => setGenome(e.target.value)} placeholder="(UCSC API)" disabled={busy} />
            <button className={browseBtn} type="button" disabled={busy}
              onClick={() => void pickFile(setGenome, [{ name: "FASTA", extensions: ["fa", "fasta", "fna"] }])}>
              …
            </button>
          </div>
        </Field>

        <Field label="Max Loci" hint="Blank = all">
          <input className={input} type="number" min={1} value={maxLoci} onChange={(e) => setMaxLoci(e.target.value)} placeholder="all" disabled={busy} />
        </Field>

        <Field label="Expression CSV" hint="CSV/BED with chr/start/stop (optional)">
          <div className="flex gap-2">
            <input className={`${input} min-w-0 flex-1`} value={exprCsv} onChange={(e) => setExprCsv(e.target.value)} placeholder="(none)" disabled={busy} />
            <button className={browseBtn} type="button" disabled={busy}
              onClick={() => void pickFile(setExprCsv, [{ name: "Table", extensions: ["csv", "tsv", "bed"] }])}>
              …
            </button>
          </div>
        </Field>

        <Field label="Output Directory">
          <input className={input} value={outDir} onChange={(e) => setOutDir(e.target.value)} disabled={busy} />
        </Field>

      </div>

      <div className="border-t border-[var(--app-border)] p-3">
        {busy ? (
          <button type="button" className={cancelBtn} onClick={() => void cancel()}>Cancel</button>
        ) : (
          <button type="button" className={runBtn} onClick={() => void run()}>Run Local Pipeline</button>
        )}
      </div>
    </div>
  );
}

// ── HPC panel ────────────────────────────────────────────────────────────────

type StepId =
  | "rmsk-query" | "dfam-count" | "prep" | "clustering" | "alignment"
  | "motif" | "go" | "expression" | "enrichment"
  | "batch" | "job-status" | "job-watch" | "retrieve";

interface StepDef {
  id: StepId;
  label: string;
  files: string[];
}

const STEP_GROUPS: { group: string; steps: StepDef[] }[] = [
  {
    group: "Data prep",
    steps: [
      { id: "rmsk-query",  label: "RMSK locus count",           files: ["te_prep.py"] },
      { id: "dfam-count",  label: "Dfam locus count",           files: ["te_prep.py"] },
      { id: "prep",        label: "Download + extract sequences", files: ["te_prep.py"] },
    ],
  },
  {
    group: "Core analysis",
    steps: [
      { id: "clustering", label: "Clustering — k-mer · UMAP · HDBSCAN", files: ["te_clustering.py", "te_prep.py"] },
      { id: "alignment",  label: "Alignment — MAFFT · CIAlign · consensus", files: ["te_alignment.py"] },
    ],
  },
  {
    group: "Enrichment",
    steps: [
      { id: "motif",      label: "Motif enrichment — JASPAR + Fisher", files: ["te_motif.py"] },
      { id: "go",         label: "GO annotation", files: ["te_go.py"] },
      { id: "expression", label: "Expression boxplots", files: ["te_expression.py"] },
      { id: "enrichment", label: "Full enrichment — motif + GO + expression", files: ["te_enrichment.py", "te_motif.py", "te_go.py", "te_expression.py"] },
    ],
  },
  {
    group: "Full pipeline",
    steps: [
      { id: "batch", label: "Submit batch job (bsub / sbatch)", files: [] },
    ],
  },
  {
    group: "Job management",
    steps: [
      { id: "job-status", label: "Check job status", files: [] },
      { id: "job-watch",  label: "Peek job output", files: [] },
    ],
  },
  {
    group: "Results",
    steps: [
      { id: "retrieve", label: "Retrieve results to local machine", files: [] },
    ],
  },
];

/** Return path as-is if absolute, otherwise use it relative to the cd'd work dir. */
function resolvePath(p: string, fallback: string): string {
  const t = p.trim();
  return t || fallback;
}

function buildCommand(step: StepId, f: FieldState, hpcHome: string, scheduler: string): string {
  const cd = `cd ${hpcHome} &&`;
  const isLsf = scheduler.toUpperCase() === "LSF";
  switch (step) {
    case "rmsk-query":
      return `${cd} python te_prep.py --build ${f.assembly} --family ${f.family || "FAMILY"} --count-only --source rmsk --rmsk-dir ${hpcHome}/rmsk`;
    case "dfam-count":
      return `${cd} python te_prep.py --build ${f.assembly} --family ${f.family || "FAMILY"} --count-only --source dfam --rmsk-dir ${hpcHome}/rmsk`;
    case "prep": {
      let cmd = `${cd} python te_prep.py --build ${f.assembly} --family ${f.family || "FAMILY"} --source ${f.annotSource}`;
      if (f.genome.trim()) cmd += ` --genome-fa ${f.genome.trim()}`;
      cmd += ` --base-dir ${resolvePath(f.outDir, "te_data")}`;
      return cmd;
    }
    case "clustering": {
      const outDir = resolvePath(f.outDir, "results");
      const family = f.family || "FAMILY";
      const clusterParts = [
        `python te_clustering.py`,
        `--kmer ${f.kmer}`,
        `--out-dir ${outDir}`,
        `--family ${family}`,
        `--n-neighbors ${f.nNeighbors || "30"}`,
        `--min-dist ${f.minDist || "0.0"}`,
        `--min-samples ${f.minSamples || "7"}`,
        `--random-state ${f.randomState || "42"}`,
        `--pca-dims ${f.pcaDims || "40"}`,
        `--n-epochs ${f.nEpochs || "120"}`,
      ];
      if (f.minClusterSize) clusterParts.push(`--min-cluster-size ${f.minClusterSize}`);
      if (f.fetchFromUcsc) {
        // Chain: te_prep.py (fetch) → te_clustering.py (cluster)
        const seqDir = `${outDir}/${family.toLowerCase()}_seqs`;
        const prepCmd = `python te_prep.py --build ${f.assembly} --family ${family} --base-dir ${seqDir}`;
        // te_prep.py writes to {base-dir}/{family}_analysis_results/clustered_data.csv
        clusterParts.push(`--input ${seqDir}/${family.toLowerCase()}_analysis_results/clustered_data.csv`);
        if (f.outputCsv.trim()) clusterParts.push(`--output ${f.outputCsv.trim()}`);
        return `${cd} ${prepCmd} && ${clusterParts.join(" ")}`;
      }
      clusterParts.push(`--input ${resolvePath(f.inputCsv, "sequences.csv")}`);
      if (f.outputCsv.trim()) clusterParts.push(`--output ${f.outputCsv.trim()}`);
      return `${cd} ${clusterParts.join(" ")}`;
    }
    case "alignment": {
      const parts = [
        `${cd} python te_alignment.py`,
        `--input ${resolvePath(f.inputCsv, "clustered.csv")}`,
        `--out-dir ${resolvePath(f.outDir, "results")}`,
        `--family ${f.family || "FAMILY"}`,
      ];
      if (f.skipCialign) parts.push("--no-cialign");
      return parts.join(" ");
    }
    case "motif": {
      const parts = [
        `${cd} python te_motif.py`,
        `--input ${resolvePath(f.inputCsv, "clustered.csv")}`,
        `--build ${f.assembly}`,
        `--out-dir ${resolvePath(f.outDir, "results")}`,
      ];
      if (f.jasparBed.trim()) parts.push(`--jaspar-bed ${f.jasparBed.trim()}`);
      if (f.runHomer) {
        parts.push("--homer");
        if (f.homerGenome.trim()) parts.push(`--homer-genome ${f.homerGenome.trim()}`);
        parts.push(`--homer-size ${f.homerSize}`, `--homer-threads ${f.homerThreads}`);
      }
      return parts.join(" ");
    }
    case "go": {
      const parts = [
        `${cd} python te_go.py`,
        `--enrichment-dir ${resolvePath(f.enrichmentDir, "results/enrichment_results")}`,
        `--build ${f.assembly}`,
        `--out-dir ${resolvePath(f.outDir, "results")}`,
      ];
      if (f.clusteredCsv.trim()) parts.push(`--clustered-csv ${f.clusteredCsv.trim()}`);
      return parts.join(" ");
    }
    case "expression":
      return [
        `${cd} python te_expression.py`,
        `--input ${resolvePath(f.inputCsv, "clustered.csv")}`,
        `--out-dir ${resolvePath(f.outDir, "results")}`,
      ].join(" ");
    case "enrichment": {
      const parts = [
        `${cd} python te_enrichment.py`,
        `--input ${resolvePath(f.inputCsv, "clustered.csv")}`,
        `--build ${f.assembly}`,
        `--out-dir ${resolvePath(f.outDir, "results")}`,
      ];
      if (f.jasparBed.trim()) parts.push(`--jaspar-bed ${f.jasparBed.trim()}`);
      if (f.skipGo) parts.push("--skip-go");
      if (f.skipExpr) parts.push("--skip-expression");
      return parts.join(" ");
    }
    case "job-status":
      return isLsf ? "bjobs" : "squeue --me";
    case "job-watch":
      if (!f.jobId.trim()) return isLsf ? "bjobs" : "squeue --me";
      return isLsf ? `bpeek ${f.jobId.trim()}` : `squeue -j ${f.jobId.trim()} --format="%i %T %R"`;
    default:
      return "";
  }
}

interface FieldState {
  family: string;
  species: string;
  assembly: string;
  outDir: string;
  inputCsv: string;
  outputCsv: string;
  enrichmentDir: string;
  clusteredCsv: string;
  teCounts: string;
  kmer: string;
  skipCialign: boolean;
  jasparBed: string;
  runHomer: boolean;
  homerGenome: string;
  homerSize: string;
  homerThreads: string;
  skipGo: boolean;
  skipExpr: boolean;
  genome: string;
  queue: string;
  memMb: string;
  cpus: string;
  walltime: string;
  modules: string;
  jasparTabix: string;
  skipJaspar: boolean;
  skipPrimers: boolean;
  skipTsne: boolean;
  fetchFromUcsc: boolean;
  notifyEmail: string;
  // Advanced pipeline params
  primerK: string;
  topNGlobal: string;
  topNCluster: string;
  minSeqClustering: string;
  primerTimeout: string;
  fetchWorkers: string;
  pcaDims: string;
  nEpochs: string;
  randomState: string;
  nNeighbors: string;
  minDist: string;
  minClusterSize: string;
  minSamples: string;
  pThreshold: string;
  // Annotation source
  annotSource: "rmsk" | "dfam";
  // Job management
  jobId: string;
  localDir: string;
  remoteDir: string;
  maxJobs: string;
  parallelMode: boolean;
}

const defaultFields: FieldState = {
  family: "", species: "human", assembly: "hg38", outDir: "",
  inputCsv: "", outputCsv: "", enrichmentDir: "", clusteredCsv: "", teCounts: "",
  kmer: "6", skipCialign: false, fetchFromUcsc: false,
  jasparBed: "", runHomer: false, homerGenome: "", homerSize: "200", homerThreads: "4",
  skipGo: false, skipExpr: false,
  genome: "", queue: "normal", memMb: "12000", cpus: "4", walltime: "04:00",
  modules: "", jasparTabix: "", skipJaspar: false, skipPrimers: false, skipTsne: true,
  notifyEmail: "",
  primerK: "18", topNGlobal: "8", topNCluster: "5", minSeqClustering: "10",
  primerTimeout: "120", fetchWorkers: "10", pcaDims: "40", nEpochs: "120",
  randomState: "0", nNeighbors: "30", minDist: "0.0", minClusterSize: "", minSamples: "7",
  pThreshold: "0.05",
  annotSource: "rmsk",
  jobId: "", localDir: "./hpc_results", remoteDir: "",
  maxJobs: "10",
  parallelMode: false,
};

function HpcPanel() {
  const [host, setHost] = useState("");
  const [port, setPort] = useState("22");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [scheduler, setScheduler] = useState<"auto" | "lsf" | "slurm">("auto");
  const [workDir, setWorkDir] = useState("");
  const [step, setStep] = useState<StepId>("rmsk-query");
  const [fields, setFields] = useState<FieldState>(defaultFields);
  const [remoteCmd, setRemoteCmd] = useState("");
  const [countResult, setCountResult] = useState<{ family: string; build: string; source: string; count: string } | null>(null);

  const hpcConnected = useAppStore((s) => s.hpcConnected);
  const hpcHost = useAppStore((s) => s.hpcHost);
  const hpcScheduler = useAppStore((s) => s.hpcScheduler);
  const hpcHome = useAppStore((s) => s.hpcHome);
  const setHpcConnection = useAppStore((s) => s.setHpcConnection);
  const hpcJobId = useAppStore((s) => s.hpcJobId);
  const setHpcJobId = useAppStore((s) => s.setHpcJobId);
  const toggleFileViewer = useAppStore((s) => s.toggleFileViewer);

  const runPhase = useAppStore((s) => s.runPhase);
  const runId = useAppStore((s) => s.runId);
  const beginRun = useAppStore((s) => s.beginRun);
  const finishRun = useAppStore((s) => s.finishRun);
  const resetRun = useAppStore((s) => s.resetRun);
  const pushToast = useAppStore((s) => s.pushToast);
  const busy = runPhase === "running";

  const set = useCallback(<K extends keyof FieldState>(k: K, v: FieldState[K]) => {
    setFields((prev) => ({ ...prev, [k]: v }));
  }, []);

  // Auto-rebuild remote command whenever fields or step change
  useEffect(() => {
    const auto = buildCommand(step, fields, hpcHome || "~", hpcScheduler);
    setRemoteCmd(auto);
    setCountResult(null);
  }, [step, fields, hpcHome, hpcScheduler]);

  // Keep jobId field in sync with last submitted job
  useEffect(() => {
    if (hpcJobId && !fields.jobId) {
      setFields((prev) => ({ ...prev, jobId: hpcJobId }));
    }
  }, [hpcJobId, fields.jobId]);

  const connect = useCallback(async () => {
    if (!host.trim() || !user.trim()) { pushToast("Host and username are required."); return; }
    try {
      beginRun(globalThis.crypto.randomUUID());
      const raw = await pyRunFull(["hpc-connect",
        "--host", host.trim(),
        "--user", user.trim(),
        "--password", password,
        "--port", port,
        "--scheduler", scheduler,
        "--work-dir", workDir.trim(),
      ]);
      const result = raw.result as { ok?: boolean; host?: string; scheduler?: string; home?: string } | undefined;
      if (result?.ok) {
        setHpcConnection({
          host: result.host ?? host,
          scheduler: result.scheduler ?? "",
          home: result.home ?? "",
        });
        finishRun("done");
      } else {
        finishRun("errored");
      }
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
      finishRun("errored");
    }
  }, [host, port, user, password, scheduler, workDir, beginRun, finishRun, pushToast, setHpcConnection]);

  const disconnect = useCallback(async () => {
    try { await pyRun(["hpc-disconnect"]); } catch (_) {}
    setHpcConnection(null);
  }, [setHpcConnection]);

  const stepDef = STEP_GROUPS.flatMap((g) => g.steps).find((s) => s.id === step);

  const runStep = useCallback(async () => {
    try {
      beginRun(globalThis.crypto.randomUUID());
      if (step === "job-status") {
        await pyRunFull(["hpc-job-status"]);
        finishRun("done");
        return;
      }
      if (step === "job-watch") {
        const argv = ["hpc-job-peek"];
        if (fields.jobId.trim()) argv.push("--job-id", fields.jobId.trim());
        await pyRunFull(argv);
        finishRun("done");
        return;
      }
      if (stepDef && stepDef.files.length > 0) {
        await pyRun(["hpc-upload", ...stepDef.files]);
      }
      const isCountStep = step === "rmsk-query" || step === "dfam-count";
      if (isCountStep) {
        const raw = await pyRunFull(["hpc-run", "--cmd", remoteCmd.trim(), "--timeout", "3600"]);
        const stdout: string = (raw.result as Record<string, string> | undefined)?.stdout ?? "";
        // "Locus count for 'FAMILY' (BUILD, SOURCE): N,NNN  [Xs]"
        const m = stdout.match(/Locus count for '([^']+)' \(([^,]+), ([^)]+)\):\s*([\d,]+)/);
        if (m) {
          setCountResult({ family: m[1], build: m[2], source: m[3], count: m[4] });
        }
      } else {
        await pyRun(["hpc-run", "--cmd", remoteCmd.trim(), "--timeout", "3600"]);
      }
      finishRun("done");
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
      finishRun("errored");
    }
  }, [step, fields.jobId, stepDef, remoteCmd, beginRun, finishRun, pushToast]);

  const pickLocalDir = useCallback(async () => {
    try {
      const dir = await open({ multiple: false, directory: true });
      if (typeof dir === "string") set("localDir", dir);
    } catch (_) {}
  }, [set]);

  const submitBatch = useCallback(async () => {
    if (!fields.family.trim()) { pushToast("TE Family is required for batch submission."); return; }
    const params = {
      family: fields.family, species: fields.species, assembly: fields.assembly,
      genome: fields.genome, queue: fields.queue, walltime: fields.walltime,
      out_dir: fields.outDir || "results", input_csv: fields.inputCsv,
      te_counts: fields.teCounts,
      jaspar_bed: fields.jasparBed, jaspar_tabix: fields.jasparTabix,
      modules: fields.modules, notify_email: fields.notifyEmail,
      mem_mb: parseInt(fields.memMb) || 12000,
      cpus: parseInt(fields.cpus) || 4,
      kmer: parseInt(fields.kmer) || 6,
      primer_k: parseInt(fields.primerK) || 18,
      top_n_global: parseInt(fields.topNGlobal) || 8,
      top_n_cluster: parseInt(fields.topNCluster) || 5,
      min_seq_clustering: parseInt(fields.minSeqClustering) || 10,
      primer_timeout: parseInt(fields.primerTimeout) || 120,
      fetch_workers: parseInt(fields.fetchWorkers) || 10,
      pca_dims: parseInt(fields.pcaDims) || 40,
      n_epochs: parseInt(fields.nEpochs) || 120,
      random_state: parseInt(fields.randomState) || 0,
      n_neighbors: parseInt(fields.nNeighbors) || 30,
      min_dist: parseFloat(fields.minDist) || 0.0,
      min_cluster_size: fields.minClusterSize ? parseInt(fields.minClusterSize) : null,
      min_samples: parseInt(fields.minSamples) || 7,
      p_threshold: parseFloat(fields.pThreshold) || 0.05,
      skip_jaspar: fields.skipJaspar ? 1 : 0,
      skip_primers: fields.skipPrimers ? 1 : 0,
      skip_go: fields.skipGo ? 1 : 0,
      skip_tsne: fields.skipTsne ? 1 : 0,
      annot_source: fields.annotSource,
      max_jobs: parseInt(fields.maxJobs) || 10,
    };
    try {
      beginRun(globalThis.crypto.randomUUID());
      if (fields.parallelMode) {
        await hpcBatchParallel(params);
      } else {
        const raw = await pyRunFull(["hpc-batch-submit", "--params", JSON.stringify(params)]);
        const result = raw.result as { ok?: boolean; job_id?: string; output_dir?: string; work_dir?: string } | undefined;
        if (result?.ok) {
          if (result.work_dir) setHpcConnection({ host: hpcHost, scheduler: hpcScheduler, home: result.work_dir });
          if (result.job_id) {
            setHpcJobId(result.job_id);
            setFields((prev) => ({ ...prev, jobId: result.job_id! }));
            pushToast(`Job submitted — ID: ${result.job_id}`);
          }
          finishRun("done");
        } else {
          finishRun("errored");
        }
      }
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
      finishRun("errored");
    }
  }, [fields, beginRun, finishRun, pushToast, setHpcJobId, setHpcConnection, hpcHost, hpcScheduler]);

  const retrieve = useCallback(async () => {
    try {
      beginRun(globalThis.crypto.randomUUID());
      const argv = ["hpc-retrieve", "--local-dir", fields.localDir.trim() || "./hpc_results"];
      if (fields.remoteDir.trim()) argv.push("--remote-dir", fields.remoteDir.trim());
      await pyRun(argv);
      finishRun("done");
    } catch (e) {
      pushToast(e instanceof Error ? e.message : String(e));
      finishRun("errored");
    }
  }, [fields.localDir, fields.remoteDir, beginRun, finishRun, pushToast]);

  const cancel = useCallback(async () => {
    if (!runId) return;
    try { await pyCancel(runId); } catch (_) {}
    resetRun();
  }, [runId, resetRun]);

  if (!hpcConnected) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-3 py-3">
          <p className="text-xs text-[var(--app-muted)]">SSH credentials for your SLURM or LSF cluster.</p>
          <Field label="Cluster hostname">
            <input className={input} value={host} onChange={(e) => setHost(e.target.value)} placeholder="login.cluster.edu" disabled={busy} />
          </Field>
          <Field label="SSH port">
            <input className={input} type="number" value={port} onChange={(e) => setPort(e.target.value)} disabled={busy} />
          </Field>
          <Field label="Username">
            <input className={input} value={user} onChange={(e) => setUser(e.target.value)} disabled={busy} />
          </Field>
          <Field label="Password">
            <input className={input} type="password" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !busy) void connect(); }} disabled={busy} />
          </Field>
          <Field label="Scheduler" hint="Use Auto unless the cluster hides bsub/sbatch from non-interactive SSH">
            <select className={input} value={scheduler} onChange={(e) => setScheduler(e.target.value as "auto" | "lsf" | "slurm")} disabled={busy}>
              <option value="auto">Auto detect</option>
              <option value="lsf">LSF / bsub</option>
              <option value="slurm">Slurm / sbatch</option>
            </select>
          </Field>
          <Field label="Remote work dir" hint="Optional. Leave blank to use the same auto-detection as ui.py">
            <input className={input} value={workDir} onChange={(e) => setWorkDir(e.target.value)} placeholder="(auto)" disabled={busy} />
          </Field>
        </div>
        <div className="border-t border-[var(--app-border)] p-3">
          {busy ? (
            <button type="button" className={cancelBtn} onClick={() => void cancel()}>Cancel</button>
          ) : (
            <button type="button" className={runBtn} onClick={() => void connect()}>Connect</button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Connection banner */}
      <div className="flex items-center gap-2 border-b border-[var(--app-border)] bg-[var(--app-bg)] px-3 py-2 text-xs">
        <span className="h-2 w-2 rounded-full bg-green-500 shrink-0" />
        <span className="font-medium truncate">{hpcHost}</span>
        <span className="text-[var(--app-muted)] truncate">{hpcScheduler} · {hpcHome}</span>
        <button type="button" className="ml-auto shrink-0 text-[var(--app-muted)] hover:text-[var(--app-text)] transition" onClick={toggleFileViewer}>
          Browse files
        </button>
        <button type="button" className="shrink-0 text-[var(--app-muted)] hover:text-red-500 transition" onClick={() => void disconnect()}>
          Disconnect
        </button>
      </div>

      {/* Job ID badge */}
      {hpcJobId && (
        <div className="flex items-center gap-2 border-b border-[var(--app-border)] bg-green-50 dark:bg-green-950/30 px-3 py-1.5 text-xs">
          <span className="font-medium text-green-700 dark:text-green-400">Last job:</span>
          <code className="font-mono text-green-800 dark:text-green-300">{hpcJobId}</code>
          <button type="button" className="ml-auto text-[var(--app-muted)] hover:text-[var(--app-text)]"
            onClick={() => { setHpcJobId(null); setFields((p) => ({ ...p, jobId: "" })); }}>
            ×
          </button>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {/* Step selector */}
        <Field label="Pipeline step">
          <select className={input} value={step}
            onChange={(e) => setStep(e.target.value as StepId)}
            disabled={busy}>
            {STEP_GROUPS.map((g) => (
              <optgroup key={g.group} label={g.group}>
                {g.steps.map((s) => (
                  <option key={s.id} value={s.id}>{s.label}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </Field>

        {/* ── Step-specific fields ── */}

        {/* Shared: family */}
        {(["rmsk-query","dfam-count","prep","clustering","alignment","expression","enrichment","batch"] as StepId[]).includes(step) && (
          <Field label="TE Family *" hint="e.g. HERVK, THE1D-int">
            <input className={input} value={fields.family}
              onChange={(e) => set("family", e.target.value)}
              placeholder="HERVK" disabled={busy} />
          </Field>
        )}

        {/* Shared: species */}
        {(["prep","clustering","alignment","motif","go","expression","enrichment","batch"] as StepId[]).includes(step) && (
          <Field label="Species">
            <select className={input} value={fields.species}
              onChange={(e) => {
                const s = e.target.value;
                set("species", s);
                if (s === "human" && (fields.assembly === "mm10" || fields.assembly === "mm39"))
                  set("assembly", "hg38");
                if (s === "mouse" && (fields.assembly === "hg38" || fields.assembly === "hg19"))
                  set("assembly", "mm10");
              }}
              disabled={busy}>
              <option value="human">human</option>
              <option value="mouse">mouse</option>
            </select>
          </Field>
        )}

        {/* Shared: assembly */}
        {(["rmsk-query","dfam-count","prep","clustering","alignment","motif","go","expression","enrichment","batch"] as StepId[]).includes(step) && (
          <Field label="Assembly / build">
            <select className={input} value={fields.assembly}
              onChange={(e) => set("assembly", e.target.value)}
              disabled={busy}>
              <option value="hg38">hg38 (human)</option>
              <option value="hg19">hg19 (human)</option>
              <option value="mm10">mm10 (mouse)</option>
              <option value="mm39">mm39 (mouse)</option>
            </select>
          </Field>
        )}

        {/* Annotation source — prep and full-pipeline batch */}
        {(["prep","batch"] as StepId[]).includes(step) && (
          <Field label="Annotation source" hint="RMSK = UCSC RepeatMasker; Dfam = Dfam nrph hits">
            <select className={input} value={fields.annotSource}
              onChange={(e) => set("annotSource", e.target.value as "rmsk" | "dfam")}
              disabled={busy}>
              <option value="rmsk">RMSK (UCSC RepeatMasker)</option>
              <option value="dfam">Dfam (nrph hits)</option>
            </select>
          </Field>
        )}

        {/* Clustering: UCSC fetch toggle */}
        {step === "clustering" && (
          <div className="flex items-center gap-2">
            <input id="fetch-ucsc" type="checkbox" checked={fields.fetchFromUcsc}
              onChange={(e) => set("fetchFromUcsc", e.target.checked)} disabled={busy} />
            <label htmlFor="fetch-ucsc" className="text-xs">Fetch sequences from UCSC first (chains te_prep.py → te_clustering.py)</label>
          </div>
        )}

        {/* Input CSV — hidden for clustering when fetching from UCSC */}
        {(["alignment","motif","expression","enrichment"] as StepId[]).includes(step) && (
          <Field label="Input sequences CSV" hint="Remote path on cluster">
            <input className={input} value={fields.inputCsv}
              onChange={(e) => set("inputCsv", e.target.value)}
              placeholder="clustered.csv" disabled={busy} />
          </Field>
        )}
        {step === "clustering" && !fields.fetchFromUcsc && (
          <Field label="Input sequences CSV" hint="Remote path on cluster">
            <input className={input} value={fields.inputCsv}
              onChange={(e) => set("inputCsv", e.target.value)}
              placeholder="sequences.csv" disabled={busy} />
          </Field>
        )}

        {/* Output CSV (clustering only) */}
        {step === "clustering" && (
          <Field label="Output CSV (optional)" hint="Remote path — blank = overwrite input">
            <input className={input} value={fields.outputCsv}
              onChange={(e) => set("outputCsv", e.target.value)}
              placeholder="clustered.csv" disabled={busy} />
          </Field>
        )}

        {/* k-mer size + clustering params */}
        {step === "clustering" && (
          <div className="space-y-2">
            <Field label="k-mer size">
              <input className={input} type="number" min={3} max={12} value={fields.kmer}
                onChange={(e) => set("kmer", e.target.value)} disabled={busy} />
            </Field>
            <div className="flex gap-2">
              <Field label="UMAP n_neighbors">
                <input className={input} type="number" min={2} value={fields.nNeighbors}
                  onChange={(e) => set("nNeighbors", e.target.value)} disabled={busy} />
              </Field>
              <Field label="UMAP min_dist">
                <input className={input} type="number" min={0} max={1} step={0.05} value={fields.minDist}
                  onChange={(e) => set("minDist", e.target.value)} disabled={busy} />
              </Field>
            </div>
            <div className="flex gap-2">
              <Field label="HDBSCAN min_cluster_size" hint="Blank = auto">
                <input className={input} type="number" min={2} value={fields.minClusterSize}
                  onChange={(e) => set("minClusterSize", e.target.value)}
                  placeholder="auto" disabled={busy} />
              </Field>
              <Field label="HDBSCAN min_samples">
                <input className={input} type="number" min={1} value={fields.minSamples}
                  onChange={(e) => set("minSamples", e.target.value)} disabled={busy} />
              </Field>
            </div>
          </div>
        )}

        {/* Alignment: skip CIAlign */}
        {step === "alignment" && (
          <div className="flex items-center gap-2">
            <input id="skip-cialign" type="checkbox" checked={fields.skipCialign}
              onChange={(e) => set("skipCialign", e.target.checked)} disabled={busy} />
            <label htmlFor="skip-cialign" className="text-xs">Skip CIAlign</label>
          </div>
        )}

        {/* Enrichment dir (te_go only) */}
        {step === "go" && (
          <Field label="enrichment_results directory" hint="Remote path">
            <input className={input} value={fields.enrichmentDir}
              onChange={(e) => set("enrichmentDir", e.target.value)}
              placeholder="results/enrichment_results" disabled={busy} />
          </Field>
        )}

        {/* Clustered CSV (te_go optional) */}
        {step === "go" && (
          <Field label="Clustered CSV (optional)" hint="For strand/expression plots">
            <input className={input} value={fields.clusteredCsv}
              onChange={(e) => set("clusteredCsv", e.target.value)}
              placeholder="(none)" disabled={busy} />
          </Field>
        )}

        {/* JASPAR BED */}
        {(["motif","enrichment","batch"] as StepId[]).includes(step) && (
          <Field label="JASPAR BED path" hint="Override auto-download (leave blank — JASPAR 2022 is downloaded automatically from CMMT)">
            <input className={input} value={fields.jasparBed}
              onChange={(e) => set("jasparBed", e.target.value)}
              placeholder="(auto-download from CMMT)" disabled={busy} />
          </Field>
        )}

        {/* HOMER options (motif only) */}
        {step === "motif" && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <input id="run-homer" type="checkbox" checked={fields.runHomer}
                onChange={(e) => set("runHomer", e.target.checked)} disabled={busy} />
              <label htmlFor="run-homer" className="text-xs">Run HOMER known-motif enrichment</label>
            </div>
            {fields.runHomer && (
              <>
                <Field label="HOMER genome" hint="Name or FASTA path (blank = same as build)">
                  <input className={input} value={fields.homerGenome}
                    onChange={(e) => set("homerGenome", e.target.value)}
                    placeholder="hg38" disabled={busy} />
                </Field>
                <div className="flex gap-2">
                  <Field label="HOMER -size">
                    <input className={input} type="number" value={fields.homerSize}
                      onChange={(e) => set("homerSize", e.target.value)} disabled={busy} />
                  </Field>
                  <Field label="Threads">
                    <input className={input} type="number" value={fields.homerThreads}
                      onChange={(e) => set("homerThreads", e.target.value)} disabled={busy} />
                  </Field>
                </div>
              </>
            )}
          </div>
        )}

        {/* Enrichment skip flags */}
        {step === "enrichment" && (
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <input id="skip-go" type="checkbox" checked={fields.skipGo}
                onChange={(e) => set("skipGo", e.target.checked)} disabled={busy} />
              <label htmlFor="skip-go" className="text-xs">Skip GO</label>
            </div>
            <div className="flex items-center gap-2">
              <input id="skip-expr" type="checkbox" checked={fields.skipExpr}
                onChange={(e) => set("skipExpr", e.target.checked)} disabled={busy} />
              <label htmlFor="skip-expr" className="text-xs">Skip expression</label>
            </div>
          </div>
        )}

        {/* Shared: out dir */}
        {(["prep","clustering","alignment","motif","go","expression","enrichment"] as StepId[]).includes(step) && (
          <Field label="Output directory" hint="Absolute path used as-is; relative path resolved from work dir">
            <input className={input} value={fields.outDir}
              onChange={(e) => set("outDir", e.target.value)}
              placeholder="results" disabled={busy} />
          </Field>
        )}

        {/* ── BATCH JOB fields ── */}
        {step === "batch" && (
          <div className="space-y-3">
            <Field label="Genome FASTA" hint="Remote path on cluster (optional)">
              <input className={input} value={fields.genome}
                onChange={(e) => set("genome", e.target.value)}
                placeholder="(UCSC API)" disabled={busy} />
            </Field>
            <Field label="Input CSV" hint="Pre-downloaded CSV (blank = auto-load from rmsk)">
              <input className={input} value={fields.inputCsv}
                onChange={(e) => set("inputCsv", e.target.value)}
                placeholder="(auto)" disabled={busy} />
            </Field>
            <Field label="Output directory">
              <input className={input} value={fields.outDir}
                onChange={(e) => set("outDir", e.target.value)}
                placeholder="results" disabled={busy} />
            </Field>

            <div className="rounded-md border border-[var(--app-border)] p-2 space-y-2">
              <p className="text-xs font-medium text-[var(--app-muted)]">Scheduler resources</p>
              <div className="flex gap-2">
                <Field label="Queue / partition">
                  <input className={input} value={fields.queue}
                    onChange={(e) => set("queue", e.target.value)}
                    placeholder="normal" disabled={busy} />
                </Field>
                <Field label="Walltime">
                  <input className={input} value={fields.walltime}
                    onChange={(e) => set("walltime", e.target.value)}
                    placeholder="04:00" disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="Mem (MB)">
                  <input className={input} type="number" value={fields.memMb}
                    onChange={(e) => set("memMb", e.target.value)} disabled={busy} />
                </Field>
                <Field label="CPUs">
                  <input className={input} type="number" value={fields.cpus}
                    onChange={(e) => set("cpus", e.target.value)} disabled={busy} />
                </Field>
                <Field label="Max jobs" hint="Max parallel HPC jobs (for throttled clusters)">
                  <input className={input} type="number" min={1} max={50} value={fields.maxJobs}
                    onChange={(e) => set("maxJobs", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
                <input type="checkbox" checked={fields.parallelMode} disabled={busy}
                  onChange={(e) => set("parallelMode", e.target.checked)} />
                Parallel stages — alignment, motif, and primers submit as separate jobs once clustering finishes
              </label>
              <Field label="Module loads" hint="Space-separated, e.g. python/3.11 gcc/12">
                <input className={input} value={fields.modules}
                  onChange={(e) => set("modules", e.target.value)}
                  placeholder="(none)" disabled={busy} />
              </Field>
            </div>

            <div className="rounded-md border border-[var(--app-border)] p-2 space-y-2">
              <p className="text-xs font-medium text-[var(--app-muted)]">Skip flags</p>
              {[
                ["skip-jaspar", "skipJaspar", "Skip JASPAR motif"],
                ["skip-primers", "skipPrimers", "Skip primer design"],
                ["skip-go-batch", "skipGo", "Skip GO annotation"],
                ["skip-tsne", "skipTsne", "Skip t-SNE (faster)"],
              ].map(([id, key, label]) => (
                <div key={id} className="flex items-center gap-2">
                  <input type="checkbox" id={id}
                    checked={fields[key as keyof FieldState] as boolean}
                    onChange={(e) => set(key as keyof FieldState, e.target.checked as never)}
                    disabled={busy} />
                  <label htmlFor={id} className="text-xs">{label}</label>
                </div>
              ))}
            </div>

            <Field label="Notify email" hint="For job-complete notifications (requires Gmail App Password)">
              <input className={input} type="email" value={fields.notifyEmail}
                onChange={(e) => set("notifyEmail", e.target.value)}
                placeholder="you@example.com" disabled={busy} />
            </Field>

            <Field label="JASPAR tabix path" hint="bgzip+tabix-indexed .bed.gz — reusable across families">
              <input className={input} value={fields.jasparTabix}
                onChange={(e) => set("jasparTabix", e.target.value)}
                placeholder="(none)" disabled={busy} />
            </Field>

            <div className="rounded-md border border-[var(--app-border)] p-2 space-y-2">
              <p className="text-xs font-medium text-[var(--app-muted)]">Clustering &amp; analysis parameters</p>
              <div className="flex gap-2">
                <Field label="K-mer size (K)">
                  <input className={input} type="number" min={3} max={12} value={fields.kmer}
                    onChange={(e) => set("kmer", e.target.value)} disabled={busy} />
                </Field>
                <Field label="PCA dims" hint="before UMAP">
                  <input className={input} type="number" value={fields.pcaDims}
                    onChange={(e) => set("pcaDims", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="UMAP epochs">
                  <input className={input} type="number" value={fields.nEpochs}
                    onChange={(e) => set("nEpochs", e.target.value)} disabled={busy} />
                </Field>
                <Field label="Random state" hint="0 = multicore">
                  <input className={input} type="number" value={fields.randomState}
                    onChange={(e) => set("randomState", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="UMAP n_neighbors">
                  <input className={input} type="number" min={2} value={fields.nNeighbors}
                    onChange={(e) => set("nNeighbors", e.target.value)} disabled={busy} />
                </Field>
                <Field label="UMAP min_dist">
                  <input className={input} type="number" min={0} max={1} step={0.05} value={fields.minDist}
                    onChange={(e) => set("minDist", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="HDBSCAN min_cluster_size" hint="Blank = N÷5">
                  <input className={input} type="number" min={2} value={fields.minClusterSize}
                    onChange={(e) => set("minClusterSize", e.target.value)}
                    placeholder="auto" disabled={busy} />
                </Field>
                <Field label="HDBSCAN min_samples">
                  <input className={input} type="number" min={1} value={fields.minSamples}
                    onChange={(e) => set("minSamples", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="Min seqs for clustering">
                  <input className={input} type="number" value={fields.minSeqClustering}
                    onChange={(e) => set("minSeqClustering", e.target.value)} disabled={busy} />
                </Field>
                <Field label="Fetch workers">
                  <input className={input} type="number" value={fields.fetchWorkers}
                    onChange={(e) => set("fetchWorkers", e.target.value)} disabled={busy} />
                </Field>
              </div>
            </div>

            <div className="rounded-md border border-[var(--app-border)] p-2 space-y-2">
              <p className="text-xs font-medium text-[var(--app-muted)]">Primer design</p>
              <div className="flex gap-2">
                <Field label="Primer K-mer">
                  <input className={input} type="number" value={fields.primerK}
                    onChange={(e) => set("primerK", e.target.value)} disabled={busy} />
                </Field>
                <Field label="Timeout (s)">
                  <input className={input} type="number" value={fields.primerTimeout}
                    onChange={(e) => set("primerTimeout", e.target.value)} disabled={busy} />
                </Field>
              </div>
              <div className="flex gap-2">
                <Field label="Top N global">
                  <input className={input} type="number" value={fields.topNGlobal}
                    onChange={(e) => set("topNGlobal", e.target.value)} disabled={busy} />
                </Field>
                <Field label="Top N cluster">
                  <input className={input} type="number" value={fields.topNCluster}
                    onChange={(e) => set("topNCluster", e.target.value)} disabled={busy} />
                </Field>
              </div>
            </div>

            <Field label="P-value threshold" hint="Fisher exact test cutoff (0 &lt; p ≤ 1)">
              <input className={input} type="number" step="0.01" min={0.001} max={1} value={fields.pThreshold}
                onChange={(e) => set("pThreshold", e.target.value)} disabled={busy} />
            </Field>

            <Field label="te_counts path" hint="Optional pre-counted TSV on cluster">
              <input className={input} value={fields.teCounts}
                onChange={(e) => set("teCounts", e.target.value)}
                placeholder="(auto)" disabled={busy} />
            </Field>
          </div>
        )}

        {/* ── JOB WATCH ── */}
        {step === "job-watch" && (
          <Field label="Job ID" hint="Optional. Blank uses the latest job submitted by this app">
            <input className={input} value={fields.jobId}
              onChange={(e) => set("jobId", e.target.value)}
              placeholder={hpcJobId ?? "12345"} disabled={busy} />
          </Field>
        )}

        {/* ── RETRIEVE ── */}
        {step === "retrieve" && (
          <div className="space-y-3">
            <Field label="Remote results path" hint="Leave blank to auto-detect">
              <input className={input} value={fields.remoteDir}
                onChange={(e) => set("remoteDir", e.target.value)}
                placeholder={`${hpcHome}/results`} disabled={busy} />
            </Field>
            <Field label="Local output directory">
              <div className="flex gap-2">
                <input className={`${input} min-w-0 flex-1`} value={fields.localDir}
                  onChange={(e) => set("localDir", e.target.value)}
                  placeholder="./hpc_results" disabled={busy} />
                <button className={browseBtn} type="button" disabled={busy}
                  onClick={() => void pickLocalDir()}>
                  …
                </button>
              </div>
            </Field>
          </div>
        )}

        {/* Remote command preview (not for batch or retrieve) */}
        {step !== "batch" && step !== "retrieve" && step !== "job-status" && step !== "job-watch" && (
          <Field label="Remote command" hint="Auto-generated — edit if needed">
            <textarea
              className={`${input} h-20 resize-y font-mono text-xs`}
              value={remoteCmd}
              onChange={(e) => setRemoteCmd(e.target.value)}
              disabled={busy}
            />
          </Field>
        )}
      </div>

      {/* Count result box */}
      {countResult && (["rmsk-query","dfam-count"] as StepId[]).includes(step) && (
        <div className="mx-3 mb-2 rounded-md border border-[var(--app-accent)]/40 bg-[var(--app-accent)]/5 p-3 text-xs space-y-1">
          <p className="font-semibold text-[var(--app-text)]">
            {countResult.source === "dfam" ? "Dfam" : "RepeatMasker"} Query Result
          </p>
          <p className="text-[var(--app-muted)]">Family: <span className="text-[var(--app-text)] font-medium">{countResult.family}</span></p>
          <p className="text-[var(--app-muted)]">Assembly: <span className="text-[var(--app-text)] font-medium">{countResult.build}</span></p>
          <p className="text-[var(--app-muted)]">Source: <span className="text-[var(--app-text)] font-medium">{countResult.source}</span></p>
          <p className="text-base font-bold text-[var(--app-accent)]">{countResult.count} loci</p>
        </div>
      )}

      {/* Action buttons */}
      <div className="border-t border-[var(--app-border)] p-3 space-y-2">
        {busy ? (
          <button type="button" className={cancelBtn} onClick={() => void cancel()}>Cancel</button>
        ) : step === "batch" ? (
          <button type="button" className={runBtn} onClick={() => void submitBatch()}>
            {fields.parallelMode ? "Submit Parallel Stages" : "Submit Batch Job"}
          </button>
        ) : step === "retrieve" ? (
          <button type="button" className={runBtn} onClick={() => void retrieve()}>
            Retrieve Results
          </button>
        ) : (
          <button type="button" className={runBtn} onClick={() => void runStep()}>
            {step === "job-status" ? "Check Status" : step === "job-watch" ? "Peek Output" : "Run on Cluster"}
          </button>
        )}
      </div>
    </div>
  );
}

// ── Shared primitives ─────────────────────────────────────────────────────────

const input =
  "w-full rounded-md border border-[var(--app-border)] bg-[var(--app-bg)] px-2 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] disabled:opacity-50";
const browseBtn =
  "shrink-0 rounded-md border border-[var(--app-border)] px-2 py-1.5 text-xs font-medium transition hover:bg-[var(--app-bg)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)] disabled:opacity-50";
const runBtn =
  "w-full rounded-md bg-[var(--app-accent)] px-3 py-2 text-sm font-semibold text-white transition hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--app-accent)]";
const cancelBtn =
  "w-full rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm font-semibold text-red-800 transition hover:bg-red-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200";

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="block text-xs font-medium">{label}</label>
      {hint && <p className="text-xs text-[var(--app-muted)]">{hint}</p>}
      {children}
    </div>
  );
}

// ── Top-level panel ────────────────────────────────────────────────────────────

export default function ArgumentsPanel({ width }: { width: number }) {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);

  return (
    <aside className="flex shrink-0 flex-col bg-[var(--app-panel)]" style={{ width }}>
      <div className="flex shrink-0 border-b border-[var(--app-border)]">
        {(["local", "hpc"] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={`flex-1 py-2 text-xs font-semibold uppercase tracking-wide transition ${
              mode === m
                ? "border-b-2 border-[var(--app-accent)] text-[var(--app-accent)]"
                : "text-[var(--app-muted)] hover:text-[var(--app-text)]"
            }`}
          >
            {m === "local" ? "Local" : "HPC"}
          </button>
        ))}
      </div>
      {mode === "local" ? <LocalPanel /> : <HpcPanel />}
    </aside>
  );
}
