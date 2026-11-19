/**
 * 多序列折线图（纯 SVG，无第三方依赖）。
 *
 * 供「问题分析 · 单题趋势」与「多任务曲线对比」共用，保证两处观感一致：
 * 同一张图里可叠多条曲线、每个序列独立配色、hover 显示读数。
 *
 * 对外接口：window.RiMultiChart
 *   renderLineChart(opts) —— 折线图
 *   renderSparkline(opts) —— 迷你折线（无坐标轴）
 */
(function (global) {
    "use strict";

    var PALETTE = [
        "#2d6a9f", "#c2410c", "#2d6a4f", "#7c3aed",
        "#b45309", "#0891b2", "#be185d", "#4d7c0f"
    ];

    function esc(s) {
        return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
            return {
                "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
            }[c];
        });
    }

    function colorAt(index) {
        return PALETTE[index % PALETTE.length];
    }

    function isNum(v) {
        return v != null && !Number.isNaN(Number(v));
    }

    function fmtNum(v) {
        if (!isNum(v)) return "-";
        var n = Number(v);
        if (Number.isInteger(n)) return String(n);
        var abs = Math.abs(n);
        if (abs >= 1000) return n.toFixed(0);
        if (abs >= 10) return n.toFixed(2);
        return n.toFixed(3);
    }

    function fmtPct(v) {
        if (!isNum(v)) return "-";
        return (Number(v) * 100).toFixed(1) + "%";
    }

    /**
     * 渲染折线图。
     *
     * @param {object} opts
     *   host      —— 容器元素（必填）
     *   series    —— [{label, color, points:[{x, y, meta}], dashed:boolean}]（必填）
     *   xLabel    —— 横轴标题
     *   yLabel    —— 纵轴标题
     *   width     —— SVG 宽度（默认 760）
     *   height    —— SVG 高度（默认 220）
     *   yFormat   —— 纵轴刻度格式化（默认 fmtNum）
     *   tipFormat —— 读数格式化（默认 "label · x · y"）
     *   yMin/yMax —— 纵轴范围（缺省自适应）
     *   legend    —— 是否显示图例（默认 true）
     */
    function renderLineChart(opts) {
        var host = opts && opts.host;
        if (!host) return;

        var series = (opts.series || []).filter(function (s) {
            return s && (s.points || []).length;
        });
        if (!series.length) {
            host.innerHTML = '<div class="muted">没有可绘制的数据。</div>';
            return;
        }

        var w = opts.width || 760;
        var h = opts.height || 220;
        var padL = opts.padL || 58;
        var padR = opts.padR || 16;
        var padT = opts.padT || 14;
        var padB = opts.padB || 34;
        var plotW = w - padL - padR;
        var plotH = h - padT - padB;
        var yFormat = opts.yFormat || fmtNum;
        var tipFormat = opts.tipFormat || function (s, p) {
            return s.label + " · " + p.x + " · " + yFormat(p.y);
        };

        // ---- 数值范围
        var xs = [], ys = [];
        series.forEach(function (s) {
            (s.points || []).forEach(function (p) {
                if (isNum(p.x)) xs.push(Number(p.x));
                if (isNum(p.y)) ys.push(Number(p.y));
            });
        });
        if (!xs.length || !ys.length) {
            host.innerHTML = '<div class="muted">没有可绘制的数据。</div>';
            return;
        }

        var xMin = opts.xMin != null ? opts.xMin : Math.min.apply(null, xs);
        var xMax = opts.xMax != null ? opts.xMax : Math.max.apply(null, xs);
        var yMin = opts.yMin != null ? opts.yMin : Math.min.apply(null, ys);
        var yMax = opts.yMax != null ? opts.yMax : Math.max.apply(null, ys);
        if (yMax === yMin) {
            var mid = yMin;
            var span = Math.abs(mid) * 0.1 || 1;
            yMin -= span; yMax += span;
        } else {
            var padY = (yMax - yMin) * 0.12;
            yMin -= padY; yMax += padY;
        }
        if (xMax === xMin) xMax = xMin + 1;

        var xAt = function (v) {
            return padL + ((v - xMin) / (xMax - xMin)) * plotW;
        };
        var yAt = function (v) {
            return padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
        };

        // ---- 网格与坐标轴
        var grid = "";
        [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
            var tick = yMin + t * (yMax - yMin);
            var y = yAt(tick);
            grid += '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (padL + plotW)
                + '" y2="' + y.toFixed(1) + '" stroke="#e4e8ec" stroke-width="1"></line>';
            grid += '<text x="' + (padL - 8) + '" y="' + (y + 3).toFixed(1)
                + '" text-anchor="end" font-size="11" fill="#5c6b7a">'
                + esc(yFormat(tick)) + "</text>";
        });

        var xLabels = "";
        var ticks = 6;
        for (var i = 0; i <= ticks; i++) {
            var xv = xMin + (xMax - xMin) * (i / ticks);
            var x = xAt(xv);
            xLabels += '<text x="' + x.toFixed(1) + '" y="' + (padT + plotH + 16)
                + '" text-anchor="middle" font-size="11" fill="#5c6b7a">'
                + esc(fmtNum(xv)) + "</text>";
        }

        // ---- 曲线
        var body = "";
        series.forEach(function (s, si) {
            var color = s.color || colorAt(si);
            var pts = (s.points || []).filter(function (p) {
                return isNum(p.x) && isNum(p.y);
            });
            if (!pts.length) return;
            var d = pts.map(function (p, pi) {
                return (pi === 0 ? "M" : "L") + xAt(Number(p.x)).toFixed(1) + ","
                    + yAt(Number(p.y)).toFixed(1);
            }).join(" ");
            body += '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.9"'
                + (s.dashed ? ' stroke-dasharray="5 4"' : "") + '></path>';
            var dots = pts.map(function (p, pi) {
                var x = xAt(Number(p.x)).toFixed(1);
                var y = yAt(Number(p.y)).toFixed(1);
                var tip = tipFormat(s, p);
                return '<g data-s="' + si + '" data-i="' + pi + '">'
                    + '<circle cx="' + x + '" cy="' + y + '" r="10" fill="transparent"></circle>'
                    + '<circle cx="' + x + '" cy="' + y + '" r="3.6" fill="' + color
                    + '" stroke="#ffffff" stroke-width="1"><title>' + esc(tip) + "</title></circle>"
                    + "</g>";
            }).join("");
            body += dots;
        });

        var legendHtml = "";
        if (opts.legend !== false) {
            legendHtml = '<div class="ri-mc-legend">'
                + series.map(function (s, si) {
                    var color = s.color || colorAt(si);
                    return '<span class="ri-mc-legend-item">'
                        + '<span class="ri-mc-swatch" style="background:' + color + '"></span>'
                        + esc(s.label || ("序列 " + (si + 1))) + "</span>";
                }).join("")
                + "</div>";
        }

        host.innerHTML = legendHtml
            + '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="'
            + esc(opts.ariaLabel || "折线图") + '">'
            + '<rect x="0" y="0" width="' + w + '" height="' + h + '" fill="#ffffff"></rect>'
            + grid
            + '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (padL + plotW)
            + '" y2="' + (padT + plotH) + '" stroke="#5c6b7a" stroke-width="1"></line>'
            + '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + plotH)
            + '" stroke="#5c6b7a" stroke-width="1"></line>'
            + xLabels
            + (opts.xLabel ? '<text x="' + (padL + plotW / 2) + '" y="' + (h - 3)
                + '" text-anchor="middle" font-size="11" fill="#5c6b7a">'
                + esc(opts.xLabel) + "</text>" : "")
            + (opts.yLabel ? '<text x="14" y="' + (padT + plotH / 2)
                + '" text-anchor="middle" font-size="11" fill="#5c6b7a" transform="rotate(-90 14 '
                + (padT + plotH / 2) + ')">' + esc(opts.yLabel) + "</text>" : "")
            + body
            + "</svg>";

        // ---- hover 读数
        var readout = opts.readout;
        var nodes = host.querySelectorAll("g[data-s]");
        if (readout && nodes.length) {
            nodes.forEach(function (node) {
                var show = function () {
                    var si = Number(node.getAttribute("data-s"));
                    var pi = Number(node.getAttribute("data-i"));
                    var s = series[si];
                    if (!s) return;
                    var p = (s.points || [])[pi];
                    if (!p) return;
                    readout.innerHTML = tipFormat(s, p);
                };
                node.addEventListener("mouseenter", show);
                node.addEventListener("click", show);
            });
        }
    }

    /**
     * 渲染迷你折线（无坐标轴、无图例）。
     *
     * @param {object} opts
     *   host    —— 容器元素
     *   values  —— 数值数组（null 表示缺点）
     *   color   —— 颜色
     *   width / height —— 尺寸（默认 210 / 46）
     *   yMin / yMax     —— 值域（缺省自适应）
     *   tipFormat —— 每个点的 title 文本
     */
    function renderSparkline(opts) {
        var host = opts && opts.host;
        if (!host) return;
        var values = (opts.values || []).filter(function (v) {
            return isNum(v);
        });
        if (!values.length) {
            host.innerHTML = '<div class="muted" style="font-size:11px;">无数据</div>';
            return;
        }
        var w = opts.width || 210;
        var h = opts.height || 46;
        var pad = 4;
        var plotW = w - pad * 2;
        var plotH = h - pad * 2;
        var color = opts.color || colorAt(0);

        var yMin = opts.yMin != null ? opts.yMin : Math.min.apply(null, values);
        var yMax = opts.yMax != null ? opts.yMax : Math.max.apply(null, values);
        if (yMax === yMin) { yMin -= 0.02; yMax += 0.02; }

        var n = values.length;
        var xAt = function (i) { return pad + (n === 1 ? plotW / 2 : i * plotW / (n - 1)); };
        var yAt = function (v) { return pad + plotH - ((v - yMin) / (yMax - yMin)) * plotH; };

        var pts = values.map(function (v, i) {
            return xAt(i).toFixed(1) + "," + yAt(Number(v)).toFixed(1);
        }).join(" ");
        var dots = values.map(function (v, i) {
            var tip = (opts.tipFormat || fmtNum)(v, i);
            return '<circle cx="' + xAt(i).toFixed(1) + '" cy="' + yAt(Number(v)).toFixed(1)
                + '" r="2.2" fill="' + color + '"><title>' + esc(tip) + "</title></circle>";
        }).join("");

        host.innerHTML = '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="'
            + esc(opts.ariaLabel || "迷你折线") + '">'
            + '<polyline points="' + pts + '" fill="none" stroke="' + color
            + '" stroke-width="1.6"></polyline>' + dots + "</svg>";
    }

    global.RiMultiChart = {
        PALETTE: PALETTE,
        colorAt: colorAt,
        esc: esc,
        fmtNum: fmtNum,
        fmtPct: fmtPct,
        renderLineChart: renderLineChart,
        renderSparkline: renderSparkline
    };
})(window);
