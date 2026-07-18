/* Backend API base-URL management. The static site is origin-agnostic; the user
   points it at wherever webapp/api is deployed. Persisted in localStorage. */
(function () {
  const KEY = "terra_api_url";

  function defaultBase() {
    // If served from the API host itself (local dev), use same origin.
    const h = location.hostname;
    if (h === "localhost" || h === "127.0.0.1") {
      return location.port === "8000" ? location.origin : "http://localhost:8000";
    }
    return ""; // on GitHub Pages there is no default — the user must set it.
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
