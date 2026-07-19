/* Backend API base-URL management. The static site is origin-agnostic; the user
   points it at wherever webapp/api is deployed. Persisted in localStorage. */
(function () {
  const KEY = "terra_api_url";

  function defaultBase() {
    // Local dev: the API runs on :8000 alongside the static server.
    const h = location.hostname;
    if (h === "localhost" || h === "127.0.0.1") {
      return location.port === "8000" ? location.origin : "http://localhost:8000";
    }
    // Deployed: default to the current origin. When the frontend is bundled
    // with the API (Cloud Run / HF Docker Space) this is exactly right and the
    // app "just works" with no configuration. If the frontend is hosted apart
    // from the API (e.g. GitHub Pages), same-origin /api simply reports offline
    // and the ⚙︎ settings panel still lets the user point at the real backend.
    return location.origin;
  }

  window.TERRA = {
    getApiBase() {
      return (localStorage.getItem(KEY) || defaultBase()).replace(/\/+$/, "");
    },
    setApiBase(url) {
      localStorage.setItem(KEY, (url || "").trim().replace(/\/+$/, ""));
    },
    hasApiBase() {
      return !!this.getApiBase();
    },
  };
})();
