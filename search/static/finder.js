(function () {
  var SOURCE_LABELS = { blog: "Blog", notes: "Notes", tweets: "Tweets", sent_mail: "Sent Mail" };

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function truncate(text, maxLen) {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen).replace(/\s+\S*$/, "") + "…";
  }

  function formatDate(iso) {
    var d = new Date(iso);
    if (isNaN(d)) return iso || "";
    return d.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
  }

  function renderResults(data) {
    var container = document.getElementById("results");
    if (data.error) {
      container.innerHTML = '<p class="error">' + escapeHtml(data.message) + "</p>";
      return;
    }
    if (!data.groups || data.groups.length === 0) {
      container.innerHTML = '<p class="empty">' + escapeHtml(data.message || "Nothing found.") + "</p>";
      return;
    }

    var html = "";
    for (var g = 0; g < data.groups.length; g++) {
      var group = data.groups[g];
      html += '<div class="group">';
      html += '<div class="group-label">' + escapeHtml(SOURCE_LABELS[group.source] || group.source) + "</div>";
      if (group.connector_text) {
        html += '<p class="connector">' + escapeHtml(group.connector_text) + "</p>";
      }
      html += '<div class="excerpts">';
      for (var i = 0; i < group.excerpts.length; i++) {
        var ex = group.excerpts[i];
        var isLink = ex.reference && ex.reference.indexOf("http") === 0;
        var metaContent = escapeHtml(ex.title) + " · " + escapeHtml(formatDate(ex.date));
        html += '<div class="excerpt">';
        html += '<p class="excerpt-text">' + escapeHtml(truncate(ex.text, 400)) + "</p>";
        if (isLink) {
          html += '<a class="excerpt-meta" href="' + escapeHtml(ex.reference) + '" target="_blank" rel="noopener">' + metaContent + "</a>";
        } else {
          html += '<div class="excerpt-meta">' + metaContent + "</div>";
        }
        html += "</div>";
      }
      html += "</div></div>";
    }
    container.innerHTML = html;
  }

  function doSearch() {
    var input = document.getElementById("search-input");
    var question = input.value.trim();
    if (!question) return;

    var container = document.getElementById("results");
    container.innerHTML = '<p class="loading">Searching…</p>';

    fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question }),
    })
      .then(function (r) { return r.json(); })
      .then(renderResults)
      .catch(function () {
        container.innerHTML = '<p class="error">Something went wrong — try again.</p>';
      });
  }

  function init() {
    var btn = document.getElementById("search-btn");
    var input = document.getElementById("search-input");
    btn.addEventListener("click", doSearch);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") doSearch();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
