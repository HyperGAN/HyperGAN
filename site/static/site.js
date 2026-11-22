/* HyperGAN Documentation Site — Vanilla Interactive Behavior
 * Zero external CDN dependencies.
 * Features: Theme toggle, Cmd/Ctrl+K search modal, Copy buttons,
 * Tab switchers, Client-side reference filters, Mobile drawer, TOC scrollspy.
 */
(() => {
  "use strict";

  const root = document.documentElement;
  const rootPrefix = root.dataset.rootPrefix || "";

  /* ------------------------------------------------------------------------
   * 1. Dark Ink / Light Paper Theme Toggle
   * ---------------------------------------------------------------------- */
  const THEME_KEY = "hg-theme";
  function applyTheme(theme) {
    if (theme === "light") {
      root.setAttribute("data-theme", "light");
    } else {
      root.removeAttribute("data-theme");
    }
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.setAttribute("aria-pressed", theme === "light" ? "true" : "false");
      btn.setAttribute(
        "title",
        theme === "light" ? "Switch to Dark Ink theme" : "Switch to Light Paper theme"
      );
    });
  }

  document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (_) {}
      applyTheme(next);
    });
  });

  /* ------------------------------------------------------------------------
   * 2. Mobile Sidebar Drawer
   * ---------------------------------------------------------------------- */
  const sidebar = document.getElementById("site-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const menuBtn = document.querySelector("[data-sidebar-toggle]");

  function setSidebarOpen(open) {
    if (!sidebar) return;
    sidebar.classList.toggle("is-open", open);
    if (backdrop) backdrop.classList.toggle("is-open", open);
    if (menuBtn) menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  if (menuBtn && sidebar) {
    menuBtn.addEventListener("click", () => {
      setSidebarOpen(!sidebar.classList.contains("is-open"));
    });
  }
  if (backdrop) {
    backdrop.addEventListener("click", () => setSidebarOpen(false));
  }

  /* ------------------------------------------------------------------------
   * 3. Copy-to-Clipboard (works on template buttons AND rendered code blocks)
   * ---------------------------------------------------------------------- */
  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }

  function flashCopied(btn) {
    const original = btn.dataset.originalLabel || btn.textContent;
    btn.dataset.originalLabel = original;
    btn.classList.add("is-copied");
    btn.textContent = "Copied";
    window.setTimeout(() => {
      btn.classList.remove("is-copied");
      btn.textContent = original;
    }, 1500);
  }

  function resolveCopyPayload(btn) {
    if (btn.dataset.copyText) {
      return btn.dataset.copyText;
    }
    const container = btn.closest(".code-block, .code-wrapper, .install-panel, .example-run, .ref-card");
    if (container) {
      const codeEl = container.querySelector("pre code, pre, code");
      if (codeEl) return codeEl.innerText.replace(/^\$ /gm, "").trimEnd();
    }
    return "";
  }

  document.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-copy-code], [data-copy-text], .copy-btn");
    if (!btn) return;
    const text = resolveCopyPayload(btn);
    if (!text) return;
    copyText(text)
      .then(() => flashCopied(btn))
      .catch(() => {});
  });

  // Auto-attach copy buttons to bare <pre> blocks in prose if markdown renderer didn't add one
  document.querySelectorAll(".prose > pre").forEach((pre) => {
    if (pre.closest(".code-block, .code-wrapper")) return;
    const wrapper = document.createElement("div");
    wrapper.className = "code-block";
    const header = document.createElement("div");
    header.className = "code-header";
    const langSpan = document.createElement("span");
    langSpan.className = "code-lang";
    langSpan.textContent = "code";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.setAttribute("data-copy-code", "1");
    btn.textContent = "Copy";
    header.append(langSpan, btn);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.append(header, pre);
  });

  /* ------------------------------------------------------------------------
   * 4. Tab Switchers (Install Bar & Landing Code Showcase)
   * ---------------------------------------------------------------------- */
  document.querySelectorAll("[data-tabs]").forEach((group) => {
    const buttons = Array.from(group.querySelectorAll("[data-tab-target]"));
    const panels = Array.from(group.querySelectorAll("[data-tab-panel]"));

    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tabTarget;
        buttons.forEach((b) => {
          const active = b === btn;
          b.classList.toggle("is-active", active);
          b.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach((p) => {
          p.classList.toggle("is-active", p.dataset.tabPanel === target);
        });
      });
    });
  });

  /* ------------------------------------------------------------------------
   * 5. Live Client-Side Filtering (API, CLI, Schema, HNDL Ops, Examples)
   * ---------------------------------------------------------------------- */
  document.querySelectorAll("[data-filter-root]").forEach((container) => {
    const input = container.querySelector("[data-filter-input]");
    const pills = Array.from(container.querySelectorAll("[data-filter-pill]"));
    const items = Array.from(container.querySelectorAll("[data-filter-item]"));
    const groups = Array.from(container.querySelectorAll("[data-filter-group]"));
    const countEl = container.querySelector("[data-filter-count]");
    const emptyEl = container.querySelector("[data-filter-empty]");

    let activeCategory = "all";

    function updateFilter() {
      const q = (input ? input.value : "").trim().toLowerCase();
      const terms = q ? q.split(/\s+/).filter(Boolean) : [];
      let visible = 0;

      items.forEach((el) => {
        const cat = (el.dataset.filterCategory || "").toLowerCase();
        const hay = (el.dataset.filterText || el.textContent || "").toLowerCase();
        const catOk = activeCategory === "all" || cat === activeCategory;
        const textOk = terms.length === 0 || terms.every((t) => hay.includes(t));
        const show = catOk && textOk;
        el.classList.toggle("is-hidden", !show);
        if (show) visible += 1;
      });

      groups.forEach((grp) => {
        const hasVisible = grp.querySelector("[data-filter-item]:not(.is-hidden)") !== null;
        grp.classList.toggle("is-hidden", !hasVisible);
      });

      if (countEl) {
        countEl.textContent = `${visible} / ${items.length} shown`;
      }
      if (emptyEl) {
        emptyEl.classList.toggle("is-visible", visible === 0);
      }
    }

    if (input) {
      input.addEventListener("input", updateFilter);
    }

    pills.forEach((pill) => {
      pill.addEventListener("click", () => {
        activeCategory = (pill.dataset.filterPill || "all").toLowerCase();
        pills.forEach((p) => {
          const on = p === pill;
          p.classList.toggle("is-active", on);
          p.setAttribute("aria-pressed", on ? "true" : "false");
        });
        updateFilter();
      });
    });

    updateFilter();
  });

  /* ------------------------------------------------------------------------
   * 6. Right-Hand "On This Page" TOC Scrollspy
   * ---------------------------------------------------------------------- */
  const tocLinks = Array.from(document.querySelectorAll(".toc-list a[href^='#']"));
  if (tocLinks.length > 0 && "IntersectionObserver" in window) {
    const headingMap = new Map();
    tocLinks.forEach((link) => {
      const id = decodeURIComponent(link.getAttribute("href").slice(1));
      const target = document.getElementById(id);
      if (target) headingMap.set(target, link);
    });

    let activeLink = null;
    const setActive = (link) => {
      if (activeLink === link) return;
      tocLinks.forEach((a) => a.classList.remove("is-active"));
      if (link) link.classList.add("is-active");
      activeLink = link;
    };

    const observer = new IntersectionObserver(
      (entries) => {
        const intersecting = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (intersecting.length > 0) {
          setActive(headingMap.get(intersecting[0].target));
        }
      },
      { rootMargin: "-84px 0px -70% 0px", threshold: 0.01 }
    );

    headingMap.forEach((_, heading) => observer.observe(heading));
  }

  /* ------------------------------------------------------------------------
   * 7. Cmd+K / Ctrl+K / '/' Search Modal
   * ---------------------------------------------------------------------- */
  const modal = document.getElementById("search-modal");
  const searchInput = document.getElementById("search-modal-input");
  const resultsEl = document.getElementById("search-modal-results");
  const statusEl = document.getElementById("search-modal-status");

  let searchIndex = null;
  let searchLoading = null;
  let selectedIdx = 0;
  let currentMatches = [];

  const KIND_META = {
    doc: { label: "Doc", cls: "badge" },
    api: { label: "API", cls: "badge badge-signal" },
    cli: { label: "CLI", cls: "badge badge-blue" },
    schema: { label: "Schema", cls: "badge badge-lilac" },
    op: { label: "HNDL Op", cls: "badge badge-warn" },
    example: { label: "Example", cls: "badge badge-signal" },
  };

  function ensureIndex() {
    if (searchIndex) return Promise.resolve(searchIndex);
    if (searchLoading) return searchLoading;
    const url = `${rootPrefix}search-index.json`;
    searchLoading = fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        searchIndex = Array.isArray(data) ? data : [];
        return searchIndex;
      })
      .catch(() => {
        searchIndex = [];
        return searchIndex;
      });
    return searchLoading;
  }

  function scoreEntry(entry, terms, rawQuery) {
    const title = (entry.title || "").toLowerCase();
    const section = (entry.section || "").toLowerCase();
    const summary = (entry.summary || "").toLowerCase();
    const tags = (entry.tags || []).join(" ").toLowerCase();
    const headings = (entry.headings || []).join(" ").toLowerCase();
    const text = (entry.text || "").toLowerCase();

    let score = 0;
    if (title === rawQuery) score += 120;
    else if (title.startsWith(rawQuery)) score += 75;
    else if (title.includes(rawQuery)) score += 45;

    for (const term of terms) {
      let matched = false;
      if (title.includes(term)) {
        score += title.startsWith(term) ? 35 : 22;
        matched = true;
      }
      if (tags.includes(term)) {
        score += 18;
        matched = true;
      }
      if (headings.includes(term)) {
        score += 12;
        matched = true;
      }
      if (section.includes(term)) {
        score += 8;
        matched = true;
      }
      if (summary.includes(term)) {
        score += 6;
        matched = true;
      }
      if (text.includes(term)) {
        score += 3;
        matched = true;
      }
      if (!matched) return 0;
    }
    return score;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderResults(query) {
    if (!resultsEl) return;
    const raw = (query || "").trim().toLowerCase();
    if (!searchIndex) {
      resultsEl.innerHTML = `<div class="search-empty">Loading search index…</div>`;
      return;
    }

    if (!raw) {
      currentMatches = searchIndex.slice(0, 10);
      selectedIdx = 0;
      if (statusEl) statusEl.textContent = `${searchIndex.length} indexed entries · type to search`;
    } else {
      const terms = raw.split(/\s+/).filter(Boolean);
      const scored = [];
      for (const entry of searchIndex) {
        const s = scoreEntry(entry, terms, raw);
        if (s > 0) scored.push({ entry, score: s });
      }
      scored.sort((a, b) => b.score - a.score || a.entry.title.localeCompare(b.entry.title));
      currentMatches = scored.slice(0, 24).map((x) => x.entry);
      selectedIdx = 0;
      if (statusEl) {
        statusEl.textContent = `${scored.length} match${scored.length === 1 ? "" : "es"}`;
      }
    }

    if (currentMatches.length === 0) {
      resultsEl.innerHTML = `<div class="search-empty">No matches for "${escapeHtml(query)}". Try searching for an API function, CLI flag, TOML table, or HNDL op.</div>`;
      return;
    }

    resultsEl.innerHTML = currentMatches
      .map((item, idx) => {
        const km = KIND_META[item.kind] || KIND_META.doc;
        const href = `${rootPrefix}${item.url}`;
        return `
          <a class="search-result-item${idx === selectedIdx ? " is-selected" : ""}" href="${escapeHtml(href)}" data-search-idx="${idx}">
            <div class="search-result-top">
              <span class="search-result-title">${escapeHtml(item.title)}</span>
              <span class="${km.cls}">${km.label}</span>
            </div>
            <div class="search-result-section">${escapeHtml(item.section || "")}</div>
            ${item.summary ? `<div class="search-result-summary">${escapeHtml(item.summary)}</div>` : ""}
          </a>
        `;
      })
      .join("");
  }

  function updateSelection(nextIdx) {
    if (!resultsEl || currentMatches.length === 0) return;
    selectedIdx = (nextIdx + currentMatches.length) % currentMatches.length;
    const nodes = resultsEl.querySelectorAll(".search-result-item");
    nodes.forEach((node, idx) => {
      const on = idx === selectedIdx;
      node.classList.toggle("is-selected", on);
      if (on) node.scrollIntoView({ block: "nearest" });
    });
  }

  function openSearch() {
    if (!modal) return;
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    ensureIndex().then(() => renderResults(searchInput ? searchInput.value : ""));
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }

  function closeSearch() {
    if (!modal) return;
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
  }

  document.querySelectorAll("[data-search-open]").forEach((btn) => {
    btn.addEventListener("click", openSearch);
  });

  if (modal) {
    modal.addEventListener("click", (ev) => {
      if (ev.target === modal) closeSearch();
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => renderResults(searchInput.value));
    searchInput.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        updateSelection(selectedIdx + 1);
      } else if (ev.key === "ArrowUp") {
        ev.preventDefault();
        updateSelection(selectedIdx - 1);
      } else if (ev.key === "Enter") {
        const active = resultsEl && resultsEl.querySelector(".search-result-item.is-selected");
        if (active) {
          ev.preventDefault();
          window.location.href = active.getAttribute("href");
        }
      }
    });
  }

  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      if (modal && modal.classList.contains("is-open")) closeSearch();
      else openSearch();
      return;
    }
    if (ev.key === "Escape") {
      if (modal && modal.classList.contains("is-open")) {
        closeSearch();
      } else {
        setSidebarOpen(false);
      }
      return;
    }
    if (
      ev.key === "/" &&
      !ev.metaKey &&
      !ev.ctrlKey &&
      !ev.altKey &&
      !/INPUT|TEXTAREA|SELECT/.test((document.activeElement && document.activeElement.tagName) || "")
    ) {
      ev.preventDefault();
      openSearch();
    }
  });
})();
