/**
 * 全量通过率折线图（按 epoch 汇总）。
 *
 * 供任务详情页与问题分析页共用，避免两处各写一份 SVG 绘制逻辑。
 * 数据口径见 service/problem_analysis_service.py 的 summarize_pass_visits：
 * 第 k 个点 = 每道题第 k 次出现的那组 attempt，按 pass 占比算通过率。
 */
(function (global) {
    "use strict";

    function esc(s) {
        return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
            return {
                "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
            }[c];
        });
    }

    function pct(v) {
        if (v == null || Number.isNaN(v)) return "-";
        return (Number(v) * 100).toFixed(1) + "%";
    }

    function epochLine(point, nProblems) {
        if (!point || point.rate == null) return "";
        var cover = nProblems
            ? " · " + point.n_problems + "/" + nProblems + " 题"
            : " · " + point.n_problems + " 题";
        return "epoch " + point.epoch + "　" + pct(point.rate)
            + "（" + point.n_pass + "/" + point.n_attempts + "）" + cover;
    }

    function pctPoints(delta) {
        if (delta == null || Number.isNaN(delta)) return "-";
        var n = delta * 100;
        var text = (n > 0 ? "+" : "") + n.toFixed(1) + " pct";
        var cls = n > 0.05 ? "up" : (n < -0.05 ? "down" : "");
        return '<span class="' + cls + '">' + text + "</span>";
    }

    /**
     * 渲染全量通过率折线。
     *
     * @param {object} opts
     *   host      —— 图表容器元素
     *   readout   —— 读数元素（hover 时显示该 epoch 明细）
     *   note      —— 说明文字元素
     *   trend     —— build_pass_trend 的返回值
     *   width     —— SVG 宽度（默认 720）
     *   height    —— SVG 高度（默认 228）
     */
    function render(opts) {
        var host = opts.host;
        var readout = opts.readout;
        var note = opts.note;
        var trend = opts.trend || {};
        var w = opts.width || 720;
        var h = opts.height || 228;

        if (!host) return;

        var series = trend.epoch_series || [];
        var nProblems = trend.n_problems || 0;

        if (!series.length) {
            host.innerHTML = "";
            if (readout) readout.textContent = "";
            if (note) note.textContent = "还没有可以按 epoch 汇总的通过率。";
            return;
        }

        var first = series[0];
        var last = series[series.length - 1];
        var delta = (first.rate != null && last.rate != null) ? last.rate - first.rate : null;
        var summary = series.length < 2
            ? epochLine(first, nProblems)
            : epochLine(first, nProblems) + "　→　" + epochLine(last, nProblems)
                + "　变化 " + pctPoints(delta);

        if (readout) readout.innerHTML = summary;

        var partial = series.some(function (p) { return nProblems && p.n_problems < nProblems; });
        if (note) {
            note.textContent = "第 k 个点：每道题按 rollout_id 从小到大，取第 k 次出现的那组 attempt；"
                + "只保留已经出现过第 k 次的题。"
                + "通过率 = 其中 pass（reward≈1）的条数 / 该组 attempt 条数。abort、超时、失败只进分母。"
                + (partial ? " 空心点表示这一档还没覆盖全部题目。" : "");
        }

        var rates = series.map(function (p) { return Number(p.rate); })
            .filter(function (v) { return !Number.isNaN(v); });
        var yMin = Math.min.apply(null, rates);
        var yMax = Math.max.apply(null, rates);
        var padY = Math.max((yMax - yMin) * 0.22, 0.015);
        yMin = Math.max(0, yMin - padY);
        yMax = Math.min(1, yMax + padY);
        if (yMax - yMin < 0.04) {
            var mid = (yMin + yMax) / 2;
            yMin = Math.max(0, mid - 0.02);
            yMax = Math.min(1, yMin + 0.04);
        }
        if (!(yMax > yMin)) { yMin = 0; yMax = 1; }

        var padL = 48, padR = 16, padT = 16, padB = 36;
        var plotW = w - padL - padR;
        var plotH = h - padT - padB;
        var n = series.length;
        var xAt = function (i) { return padL + (n === 1 ? plotW / 2 : i * plotW / (n - 1)); };
        var yAt = function (rate) {
            return padT + plotH - ((rate - yMin) / (yMax - yMin)) * plotH;
        };

        var grid = "";
        [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
            var tick = yMin + t * (yMax - yMin);
            var y = yAt(tick);
            grid += '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (padL + plotW)
                + '" y2="' + y.toFixed(1) + '" stroke="#e4e8ec" stroke-width="1"></line>';
            grid += '<text x="' + (padL - 8) + '" y="' + (y + 3).toFixed(1)
                + '" text-anchor="end" font-size="11" fill="#5c6b7a">'
                + (tick * 100).toFixed((yMax - yMin) < 0.25 ? 1 : 0) + "%</text>";
        });

        var labelEvery = n <= 12 ? 1 : Math.ceil(n / 8);
        var xLabels = "";
        series.forEach(function (point, i) {
            if (!(i === 0 || i === n - 1 || i % labelEvery === 0)) return;
            var x = xAt(i);
            xLabels += '<line x1="' + x.toFixed(1) + '" y1="' + (padT + plotH) + '" x2="'
                + x.toFixed(1) + '" y2="' + (padT + plotH + 4) + '" stroke="#5c6b7a"></line>';
            xLabels += '<text x="' + x.toFixed(1) + '" y="' + (padT + plotH + 18)
                + '" text-anchor="middle" font-size="11" fill="#5c6b7a">' + point.epoch + "</text>";
        });

        var pts = series.map(function (point, i) {
            return xAt(i).toFixed(1) + "," + yAt(point.rate).toFixed(1);
        }).join(" ");
        var base = (padT + plotH).toFixed(1);
        var area = n > 1
            ? xAt(0).toFixed(1) + "," + base + " " + pts + " " + xAt(n - 1).toFixed(1) + "," + base
            : "";

        var dots = series.map(function (point, i) {
            var full = !nProblems || point.n_problems >= nProblems;
            var x = xAt(i).toFixed(1);
            var y = yAt(point.rate).toFixed(1);
            var fill = full ? "#2d6a9f" : "#ffffff";
            var title = "epoch " + point.epoch + " · " + pct(point.rate) + " · "
                + point.n_pass + "/" + point.n_attempts + " · "
                + point.n_problems + "/" + (nProblems || point.n_problems) + " 题";
            return '<g data-i="' + i + '">'
                + '<circle cx="' + x + '" cy="' + y + '" r="11" fill="transparent"></circle>'
                + '<circle cx="' + x + '" cy="' + y + '" r="4.2" fill="' + fill
                + '" stroke="#2d6a9f" stroke-width="1.6"><title>' + esc(title) + "</title></circle>"
                + "</g>";
        }).join("");

        host.innerHTML = '<svg viewBox="0 0 ' + w + " " + h
            + '" role="img" aria-label="按 epoch 汇总的全量题目通过率">'
            + '<rect x="0" y="0" width="' + w + '" height="' + h + '" fill="#ffffff"></rect>'
            + grid
            + '<line x1="' + padL + '" y1="' + base + '" x2="' + (padL + plotW)
            + '" y2="' + base + '" stroke="#5c6b7a" stroke-width="1"></line>'
            + '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + base
            + '" stroke="#5c6b7a" stroke-width="1"></line>'
            + xLabels
            + '<text x="' + (padL + plotW / 2) + '" y="' + (h - 4)
            + '" text-anchor="middle" font-size="11" fill="#5c6b7a">epoch</text>'
            + '<text x="14" y="' + (padT + plotH / 2)
            + '" text-anchor="middle" font-size="11" fill="#5c6b7a" transform="rotate(-90 14 '
            + (padT + plotH / 2) + ')">通过率</text>'
            + (area ? '<polygon points="' + area + '" fill="#2d6a9f" opacity="0.08"></polygon>' : "")
            + (n > 1 ? '<polyline points="' + pts + '" fill="none" stroke="#2d6a9f" stroke-width="2"></polyline>' : "")
            + dots
            + "</svg>";

        host.addEventListener("mouseleave", function () {
            if (readout) readout.innerHTML = summary;
        });
        host.querySelectorAll("g[data-i]").forEach(function (node) {
            var show = function () {
                var point = series[Number(node.getAttribute("data-i"))];
                if (readout) readout.innerHTML = epochLine(point, nProblems);
            };
            node.addEventListener("mouseenter", show);
            node.addEventListener("click", show);
        });
    }

    global.RiEpochChart = { render: render };
})(window);
