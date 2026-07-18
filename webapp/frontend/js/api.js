/* Fetch wrappers around the TERRA Web API. */
(function () {
  function base() {
    const b = window.TERRA.getApiBase();
    if (!b) throw new Error("No API URL set. Click ⚙︎ and enter your backend URL.");
    return b;
  }

  async function asJson(res) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && (body.detail || body.error) ? (body.detail || body.error) : res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }

  window.API = {
    async health() {
      const res = await fetch(base() + "/api/health", { method: "GET" });
      return asJson(res);
    },
    async inspect(file) {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(base() + "/api/inspect", { method: "POST", body: fd });
      return asJson(res);
    },
    async cluster(formData) {
      const res = await fetch(base() + "/api/cluster", { method: "POST", body: formData });
      return asJson(res);
    },
    async guides(formData) {
      const res = await fetch(base() + "/api/guides", { method: "POST", body: formData });
      return asJson(res);
    },
    async job(jobId) {
      const res = await fetch(base() + "/api/jobs/" + jobId, { method: "GET" });
      return asJson(res);
    },
    fileUrl(jobId, name) {
      return base() + "/api/files/" + jobId + "/" + encodeURIComponent(name);
    },
  };
})();
