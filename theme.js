/* =========================================================
   DataLake — theme controller
   Load in <head> (not deferred) so the saved theme applies
   before first paint.
   - [data-theme-menu]   → a palette button with a popover picker
   - [data-theme-picker] → an inline picker (e.g. Settings page)
   Window event 'dl-themechange' fires after every change.
========================================================= */
(function(){
  'use strict';

  // preview colours: [background, sidebar/surface, text line, accent]
  var THEMES = [
    { id:'system',   name:'System',   scheme:null },
    { id:'light',    name:'Light',    scheme:'light', p:['#FFFFFF','#F4F4F5','#0A0A0A','#0B6BCB'] },
    { id:'dark',     name:'Dark',     scheme:'dark',  p:['#0A0A0B','#1B1B1F','#EDEDED','#4DA3FF'] },
    { id:'midnight', name:'Midnight', scheme:'dark',  p:['#0B1020','#1C2744','#E6EAF5','#7C9BFF'] },
    { id:'forest',   name:'Forest',   scheme:'dark',  p:['#0C1410','#1D2F26','#E4EFE8','#5FD49B'] },
    { id:'graphite', name:'Graphite', scheme:'dark',  p:['#161616','#2A2A2A','#F2F2F2','#F5A524'] },
    { id:'sand',     name:'Sand',     scheme:'light', p:['#FBF8F3','#EFE8DC','#1C1917','#C2410C'] },
    { id:'ocean',    name:'Ocean',    scheme:'light', p:['#F5F9FC','#E7F0F7','#0B1B2B','#0E7490'] },
    { id:'rose',     name:'Rose',     scheme:'light', p:['#FFF8F9','#FBE9ED','#1F1016','#BE185D'] }
  ];
  var BY_ID = {}; THEMES.forEach(function(t){ BY_ID[t.id] = t; });
  var KEY = 'dl-theme';
  var root = document.documentElement;
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function read(){ try { var v = localStorage.getItem(KEY); return BY_ID[v] ? v : 'light'; } catch(e){ return 'light'; } }
  function write(v){ try { localStorage.setItem(KEY, v); } catch(e){} }
  function resolvedId(){
    var c = read();
    if (c === 'system') return (mq && mq.matches) ? 'dark' : 'light';
    return c;
  }

  function apply(){
    var t = BY_ID[resolvedId()];
    root.setAttribute('data-theme', t.id);
    root.setAttribute('data-scheme', t.scheme);
    try { window.dispatchEvent(new CustomEvent('dl-themechange', { detail: { choice: read(), theme: t.id, scheme: t.scheme } })); } catch(e){}
  }

  function setTheme(id){ if (!BY_ID[id]) return; write(id); apply(); sync(); }

  apply();
  if (mq){
    var onSys = function(){ if (read() === 'system'){ apply(); sync(); } };
    if (mq.addEventListener) mq.addEventListener('change', onSys); else if (mq.addListener) mq.addListener(onSys);
  }
  window.addEventListener('storage', function(e){ if (e.key === KEY){ apply(); sync(); } });

  /* ---------- UI ---------- */
  var ICON_PALETTE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 1 1 10-10c0 2.2-1.8 3-3.5 3H16a2 2 0 0 0-1.5 3.3A2 2 0 0 1 12 22z"/><circle cx="7.5" cy="10.5" r="1"/><circle cx="12" cy="7" r="1"/><circle cx="16.5" cy="10.5" r="1"/></svg>';
  var ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';

  function previewHtml(t){
    function mini(p){
      return '<div class="p-side" style="background:' + p[1] + '"></div>' +
             '<div class="p-card" style="background:' + p[1] + '"></div>' +
             '<div class="p-line" style="background:' + p[2] + ';opacity:.7"></div>' +
             '<div class="p-dot" style="background:' + p[3] + '"></div>';
    }
    if (t.id === 'system'){
      var l = BY_ID.light.p, d = BY_ID.dark.p;
      return '<div class="tm-prev split"><div style="background:' + l[0] + '">' + mini(l) + '</div><div style="background:' + d[0] + '">' + mini(d) + '</div></div>';
    }
    return '<div class="tm-prev" style="background:' + t.p[0] + '">' + mini(t.p) + '</div>';
  }

  function buildGrid(wide){
    var grid = document.createElement('div');
    grid.className = 'tm-grid' + (wide ? ' wide' : '');
    grid.setAttribute('role', 'group'); grid.setAttribute('aria-label', 'Theme');
    THEMES.forEach(function(t){
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'tm-opt'; b.setAttribute('data-theme-id', t.id);
      b.setAttribute('aria-label', t.name + ' theme');
      b.innerHTML = previewHtml(t) + '<span class="tm-name"><span>' + t.name + '</span></span>';
      b.addEventListener('click', function(){ setTheme(t.id); });
      grid.appendChild(b);
    });
    return grid;
  }

  var menus = [];
  function buildMenu(host){
    if (host.__dl) return; host.__dl = true;
    var wrap = document.createElement('div'); wrap.className = 'tm';
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'tm-btn'; btn.title = 'Theme';
    btn.setAttribute('aria-label', 'Change theme'); btn.setAttribute('aria-haspopup', 'true'); btn.setAttribute('aria-expanded', 'false');
    btn.innerHTML = ICON_PALETTE;
    var pop = document.createElement('div'); pop.className = 'tm-pop';
    pop.setAttribute('role', 'dialog'); pop.setAttribute('aria-label', 'Theme');
    var lbl = document.createElement('div'); lbl.className = 'tm-label'; lbl.textContent = 'Theme';
    pop.appendChild(lbl); pop.appendChild(buildGrid(false));
    wrap.appendChild(btn); wrap.appendChild(pop); host.appendChild(wrap);
    btn.addEventListener('click', function(e){
      e.stopPropagation();
      var open = !wrap.classList.contains('open');
      closeAll();
      if (open){ wrap.classList.add('open'); btn.setAttribute('aria-expanded', 'true'); }
    });
    pop.addEventListener('click', function(e){ e.stopPropagation(); });
    menus.push(wrap);
  }
  function buildPicker(host){
    if (host.__dl) return; host.__dl = true;
    host.appendChild(buildGrid(true));
  }
  function closeAll(){
    menus.forEach(function(m){ m.classList.remove('open'); var b = m.querySelector('.tm-btn'); if (b) b.setAttribute('aria-expanded', 'false'); });
  }
  function sync(){
    var c = read();
    document.querySelectorAll('.tm-opt').forEach(function(b){
      var on = b.getAttribute('data-theme-id') === c;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      var name = b.querySelector('.tm-name');
      var check = name.querySelector('svg');
      if (on && !check) name.insertAdjacentHTML('beforeend', ICON_CHECK);
      if (!on && check) check.remove();
    });
  }

  function init(){
    document.querySelectorAll('[data-theme-menu]').forEach(buildMenu);
    document.querySelectorAll('[data-theme-picker]').forEach(buildPicker);
    sync();
    document.addEventListener('click', closeAll);
    document.addEventListener('keydown', function(e){ if (e.key === 'Escape') closeAll(); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

  window.DLTheme = {
    themes: THEMES.map(function(t){ return { id: t.id, name: t.name }; }),
    get: function(){ return { choice: read(), theme: resolvedId(), scheme: BY_ID[resolvedId()].scheme }; },
    set: setTheme,
    cssVar: function(name){ return getComputedStyle(root).getPropertyValue(name).trim(); }
  };
})();
