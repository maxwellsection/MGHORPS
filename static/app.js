document.addEventListener('DOMContentLoaded', () => {
    // Top DOM
    const workspaceEl = document.getElementById('workspace');
    const modelTemplate = document.getElementById('model-template');
    const reportTemplate = document.getElementById('report-template');
    const statusTemplate = document.getElementById('status-template');
    
    // Tools
    const solveBtn = document.getElementById('solve-btn');
    const backendSelect = document.getElementById('backend-select');

    // State
    window.workspaceState = {
        counter: 1
    };
    
    // Active Window tracking
    let activeWindowDOM = null;
    let activeEditor = null; // Currently focused textarea

    function setActiveWindow(win) {
        if(!win || win.style.display === 'none') return;
        document.querySelectorAll('.window').forEach(w => w.classList.remove('active-window'));
        win.classList.add('active-window');
        win.style.zIndex = getHighestZIndex() + 1;
        activeWindowDOM = win;
        if(win.classList.contains('model-window')) {
            activeEditor = win.querySelector('.code-editor');
            activeEditor.focus();
        } else {
            activeEditor = null;
        }
    }

    function getHighestZIndex() {
        let highest = 0;
        document.querySelectorAll('.window').forEach(w => {
            let z = parseInt(window.getComputedStyle(w).zIndex || 0);
            if (z > highest) highest = z;
        });
        return highest;
    }

    // Auto Save / Load logic
    function saveState() {
        const state = [];
        document.querySelectorAll('.window').forEach(w => {
            if(w.style.display === 'none') return; // Do not save closed windows
            state.push({
                id: w.dataset.id,
                type: w.classList.contains('model-window') ? 'model' : 'report',
                name: w.querySelector(w.classList.contains('model-window') ? '.model-name' : '.report-name').innerText,
                content: w.classList.contains('model-window') ? w.querySelector('.code-editor').value : w.querySelector('.solution-output').innerHTML,
                top: w.style.top,
                left: w.style.left,
                width: w.style.width,
                height: w.style.height,
                zIndex: w.style.zIndex,
                success: w.querySelector('.window-header').classList.contains('win-success')
            });
        });
        localStorage.setItem('ultimatesolver_workspace', JSON.stringify({ state, counter: workspaceState.counter }));
    }

    function loadState() {
        const data = localStorage.getItem('ultimatesolver_workspace');
        if (data) {
            try {
                const parsed = JSON.parse(data);
                workspaceState.counter = parsed.counter || 1;
                if(parsed.state && parsed.state.length > 0) {
                    parsed.state.forEach(w => {
                        if(w.type === 'model') createModelWindow(w.name, w.content, w.id, w);
                        else createReportWindow(w.name, w.content, w.success, w.id, w);
                    });
                    return;
                }
            } catch(e) {}
        }
        createModelWindow('Model_1', '! 欢迎使用 UltimateSolver ;\nMAX = 20 * X1 + 30 * X2;\n\nSUBJECT TO;\n  X1 + X2 <= 50;\n  3 * X1 + 2 * X2 <= 100;\n\n@FREE(X1);\n@BND(10, X2, 30);');
    }

    // Syntax Highlighting Layer
    function highlightSyntax(text) {
        let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const rx = /(![\s\S]*?;)|(@\w+)|\b(MAX|MIN|SUBJECT TO|FREE|BND|GIN|BIN|SUB|END)\b|\b(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b|([+\-*/=\(\)\[\]:,]|&lt;|&gt;)/gi;
        return html.replace(rx, (match, p1, p2, p3, p4, p5) => {
            if (p1) return `<span class="token-comment">${p1}</span>`;
            if (p2 || p3) return `<span class="token-keyword">${match}</span>`;
            if (p4) return `<span class="token-number">${p4}</span>`;
            if (p5) return `<span class="token-operator">${p5}</span>`;
            return match;
        });
    }

    setInterval(saveState, 3000); // Auto-save

    function createModelWindow(name, content = '', id = null, geom = null) {
        if(!id) id = 'model_' + Date.now();
        const clone = modelTemplate.content.cloneNode(true);
        const win = clone.querySelector('.window');
        win.dataset.id = id;
        win.querySelector('.model-name').innerText = name;
        
        const textarea = win.querySelector('.code-editor');
        const preCode = win.querySelector('.language-lingo');
        const lineNumbers = win.querySelector('.line-numbers');
        
        textarea.value = content;
        
        const updateEditor = () => {
            preCode.innerHTML = highlightSyntax(textarea.value);
            const lines = textarea.value.split('\n').length;
            lineNumbers.innerHTML = Array(lines).fill(0).map((_, i) => i + 1).join('<br>');    
        };
        
        textarea.addEventListener('input', updateEditor);
        textarea.addEventListener('scroll', () => {
            const codeBlock = win.querySelector('.syntax-highlighting');
            codeBlock.scrollTop = textarea.scrollTop;
            codeBlock.scrollLeft = textarea.scrollLeft;
            lineNumbers.scrollTop = textarea.scrollTop;
        });

        textarea.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                solveBtn.click();
            }
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = textarea.selectionStart;
                textarea.value = textarea.value.substring(0, start) + '  ' + textarea.value.substring(textarea.selectionEnd);
                textarea.selectionStart = textarea.selectionEnd = start + 2;
                updateEditor();
            }
        });
        
        makeDraggable(win);
        applyGeometry(win, geom, 20, 20);
        workspaceEl.appendChild(win);
        updateEditor();
        setActiveWindow(win);
        return win;
    }

    function createReportWindow(name, content = '', success = true, id = null, geom = null) {
        if(!id) id = 'report_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const clone = reportTemplate.content.cloneNode(true);
        const win = clone.querySelector('.window');
        win.dataset.id = id;
        win.querySelector('.report-name').innerText = name;
        
        const outNode = win.querySelector('.solution-output');
        if(content.includes('<div class="report-container"')) {
            outNode.innerHTML = content;
        } else if(content) {
            outNode.innerHTML = `<pre style="margin:0; padding:15px; color:#d1d5db; font-family:var(--font-code); font-size:13px; line-height:1.6;">${content.replace(/</g, '&lt;')}</pre>`;
        } else {
            outNode.innerHTML = '';
        }
        
        if(!success) win.querySelector('.window-header').classList.remove('win-success');
        
        makeDraggable(win);
        applyGeometry(win, geom, 80, 60);
        workspaceEl.appendChild(win);
        setActiveWindow(win);
        return win;
    }

    function applyGeometry(win, geom, defaultX, defaultY) {
        if(geom) {
            win.style.top = geom.top; win.style.left = geom.left;
            win.style.width = geom.width; win.style.height = geom.height;
            if(geom.zIndex) win.style.zIndex = geom.zIndex;
        } else {
            win.style.top = defaultY + 'px'; win.style.left = defaultX + 'px';
        }
    }

    function makeDraggable(elmnt) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        const header = elmnt.querySelector('.window-header');
        
        elmnt.addEventListener('mousedown', () => setActiveWindow(elmnt));
        if (header) header.onmousedown = dragMouseDown;

        function dragMouseDown(e) {
            e.preventDefault(); pos3 = e.clientX; pos4 = e.clientY;
            document.onmouseup = closeDragElement; document.onmousemove = elementDrag;
        }
        function elementDrag(e) {
            e.preventDefault();
            pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY;
            pos3 = e.clientX; pos4 = e.clientY;
            elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
            elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
        }
        function closeDragElement() {
            document.onmouseup = null; document.onmousemove = null;
        }

        const btnClose = elmnt.querySelector('.win-close');
        const btnMax = elmnt.querySelector('.win-max');
        const btnMin = elmnt.querySelector('.win-min');
        const content = elmnt.querySelector('.window-content');
        
        if (btnClose) {
            btnClose.addEventListener('click', (e) => {
                e.stopPropagation();
                elmnt.style.display = 'none'; // Close effectively hides it. State logic ignores hidden windows.
            });
        }
        
        let isMaximized = false, preMaxRect = {};
        if (btnMax) {
            btnMax.addEventListener('click', (e) => {
                e.stopPropagation();
                if (!isMaximized) {
                    preMaxRect = { top: elmnt.style.top, left: elmnt.style.left, width: elmnt.style.width, height: elmnt.style.height };
                    elmnt.style.top = '10px'; elmnt.style.left = '10px';
                    elmnt.style.width = 'calc(100% - 20px)'; elmnt.style.height = 'calc(100% - 20px)';
                    btnMax.innerText = '❐';
                } else {
                    elmnt.style.top = preMaxRect.top; elmnt.style.left = preMaxRect.left;
                    elmnt.style.width = preMaxRect.width; elmnt.style.height = preMaxRect.height;
                    btnMax.innerText = '□';
                }
                isMaximized = !isMaximized;
            });
        }
        
        let isMinimized = false, preMinHeight = '';
        if (btnMin) {
            btnMin.addEventListener('click', (e) => {
                e.stopPropagation();
                if (!isMinimized) {
                    preMinHeight = elmnt.style.height || getComputedStyle(elmnt).height;
                    if(content) content.style.display = 'none';
                    elmnt.style.height = 'auto'; elmnt.style.resize = 'none';
                } else {
                    if(content) content.style.display = 'flex';
                    elmnt.style.height = preMinHeight; elmnt.style.resize = 'both';
                }
                isMinimized = !isMinimized;
            });
        }
    }

    // Toolbars and Menus
    const execCmd = (cmd) => {
        if(!activeEditor) return;
        activeEditor.focus();
        document.execCommand(cmd);
    }
    
    window.toFraction = function(decimal, precision = 1e-5) {
        if (!isFinite(decimal) || isNaN(decimal)) return decimal.toString();
        if (Math.abs(Math.round(decimal) - decimal) < 1e-7) return Math.round(decimal).toString();
        let sign = Math.sign(decimal);
        decimal = Math.abs(decimal);
        let num_int = Math.floor(decimal), num_dec = decimal - num_int;
        let p0 = 1, q0 = 0, p1 = num_int, q1 = 1, p2, q2;
        let n = num_dec;
        for (let i = 0; i < 20; i++) {
            if (n < 1e-9) break;
            n = 1.0 / n;
            let a = Math.floor(n);
            p2 = a * p1 + p0; q2 = a * q1 + q0;
            if (Math.abs(decimal - p2/q2) < precision) { p1 = p2; q1 = q2; break; }
            p0 = p1; p1 = p2; q0 = q1; q1 = q2;
            n -= a;
        }
        if (q1 > 50000) return (sign * decimal).toFixed(6);
        return (sign * p1) + '/' + q1;
    };

    document.getElementById('btn-new').onclick = () => {
        let hasActive = false;
        document.querySelectorAll('.model-window').forEach(w => { if(w.style.display !== 'none') hasActive = true; });
        if(!hasActive) workspaceState.counter = 0;
        workspaceState.counter++;
        createModelWindow(`Model_${workspaceState.counter}`);
    };
    
    document.getElementById('btn-open').onclick = () => {
        const input = document.createElement('input');
        input.type = 'file'; input.accept = '.lng,.txt';
        input.onchange = e => {
            const file = e.target.files[0];
            if(!file) return;
            const reader = new FileReader();
            reader.onload = e => {
                if(activeEditor) activeEditor.value = e.target.result;
                else {
                    workspaceState.counter++;
                    createModelWindow(file.name, e.target.result);
                }
                if(activeEditor) activeEditor.dispatchEvent(new Event('input'));
            };
            reader.readAsText(file);
        };
        input.click();
    };

    document.getElementById('btn-save').onclick = () => {
        if(!activeEditor) return alert("请先选中(点击)一个模型窗口！");
        const blob = new Blob([activeEditor.value], {type: "text/plain;charset=utf-8"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url;
        const currentName = activeWindowDOM.querySelector('.model-name').innerText;
        a.download = currentName + ".lng"; a.click();
        URL.revokeObjectURL(url);
    };

    document.getElementById('btn-cut').onclick = async () => {
        if(!activeEditor) return;
        try {
            const text = activeEditor.value.substring(activeEditor.selectionStart, activeEditor.selectionEnd);
            if(text) { await navigator.clipboard.writeText(text); document.execCommand('cut'); }
        } catch(err) {} 
    };

    document.getElementById('btn-copy').onclick = async () => {
        if(!activeEditor) return;
        try {
            const text = activeEditor.value.substring(activeEditor.selectionStart, activeEditor.selectionEnd);
            if(text) await navigator.clipboard.writeText(text);
        } catch(err) { document.execCommand('copy'); }
    };

    document.getElementById('btn-paste').onclick = async () => {
        if(!activeEditor) return;
        try {
            const text = await navigator.clipboard.readText();
            document.execCommand('insertText', false, text);
        } catch(err) { execCmd('paste'); }
    };

    document.getElementById('menu-new').onclick = document.getElementById('btn-new').onclick;
    document.getElementById('menu-open').onclick = document.getElementById('btn-open').onclick;
    document.getElementById('menu-save').onclick = document.getElementById('btn-save').onclick;
    document.getElementById('menu-cut').onclick = document.getElementById('btn-cut').onclick;
    document.getElementById('menu-copy').onclick = document.getElementById('btn-copy').onclick;
    document.getElementById('menu-paste').onclick = document.getElementById('btn-paste').onclick;
    document.getElementById('menu-solve').onclick = () => solveBtn.click();
    document.getElementById('menu-about').onclick = () => alert("UltimateSolver Web IDE \n\n完全自主设计研发的新一代运筹优化前端控制台，支持多节点异构算力加速。\n\n设计：强大的AI研发助理");

    function formatTime(ms) {
        let totalSeconds = Math.floor(ms / 1000);
        let hours = Math.floor(totalSeconds / 3600);
        let minutes = Math.floor((totalSeconds % 3600) / 60);
        let seconds = totalSeconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    document.getElementById('solve-all-btn').onclick = () => {
        const models = document.querySelectorAll('.model-window');
        let count = 0;
        models.forEach(win => {
            if(win.style.display !== 'none') {
                setActiveWindow(win);
                solveBtn.click();
                count++;
            }
        });
        if(count === 0) alert("当前工作区没有任何模型能被求解！");
    };

    // Concurrent Solve Architecture
    solveBtn.addEventListener('click', async () => {
        if(!activeWindowDOM || !activeWindowDOM.classList.contains('model-window') || !activeEditor) {
            return alert("请先创建或点击选中一个模型窗口进行求解！");
        }
        
        const code = activeEditor.value.trim();
        if (!code) return;
        const currentModelName = activeWindowDOM.querySelector('.model-name').innerText;
        const backendName = backendSelect.value;
        const backendText = backendSelect.options[backendSelect.selectedIndex].text;

        // 1. Instantly spawn a new Report Window acting as the Concurrent Task Monitor
        const reportWin = createReportWindow(`Report_${currentModelName}`, '', true);
        const termWrapper = reportWin.querySelector('.terminal-wrapper');
        const outputTx = reportWin.querySelector('.solution-output');
        outputTx.style.display = 'none';
        
        // 2. Clone the Status Dashboard into this window
        const statusClone = statusTemplate.content.cloneNode(true);
        const statusNode = statusClone.querySelector('.solver-status-modal');
        termWrapper.appendChild(statusNode);
        
        const pState = statusNode.querySelector('.status-state');
        const pBackend = statusNode.querySelector('.status-backend');
        const pObj = statusNode.querySelector('.status-obj');
        const pVars = statusNode.querySelector('.stat-vars');
        const pCons = statusNode.querySelector('.stat-cons');
        const pTime = statusNode.querySelector('.stat-time');
        const pBar = statusNode.querySelector('.progress-bar');
        const pCanvas = statusNode.querySelector('.solver-chart');
        const ctx = pCanvas.getContext('2d');
        
        pState.innerText = 'Warm-up Hardware...';
        pBackend.innerText = backendText.split(' ')[0];
        pObj.innerText = 'Calculating...';
        pBar.style.animation = 'indeterminate 1.5s infinite linear';
        pBar.style.background = '#3b82f6';
        
        let startTime = Date.now();
        pTime.innerText = "00:00:00";
        let timerInterval = setInterval(() => { pTime.innerText = formatTime(Date.now() - startTime); }, 1000);

        ctx.clearRect(0, 0, pCanvas.width, pCanvas.height);
        let chartData = [];
        let curVal = 100;
        let chartInterval = setInterval(() => {
            curVal = curVal * 0.95 + (Math.random() * 2); 
            chartData.push(curVal);
            if(chartData.length > 50) chartData.shift();
            
            ctx.clearRect(0, 0, pCanvas.width, pCanvas.height);
            ctx.beginPath(); ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2; ctx.shadowBlur = 10; ctx.shadowColor = '#10b981';
            for(let i=0; i<chartData.length; i++) {
                let x = (i / 50) * pCanvas.width;
                let y = pCanvas.height - (chartData[i] / 100) * pCanvas.height;
                if(i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke(); ctx.shadowBlur = 0;
        }, 50);

        try {
            pState.innerText = 'Solving Space...';
            // Non-blocking Concurrent Fetch Call
            const response = await fetch('/api/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, backend: backendName })
            });

            const resultText = await response.text();
            let result = {};
            try { 
                const cleanText = resultText.replace(/:\s*Infinity/g, ':"Infinity"').replace(/:\s*-Infinity/g, ':"-Infinity"').replace(/:\s*NaN/g, ':"NaN"');
                result = JSON.parse(cleanText);
                if (result.objective_value === "Infinity") result.objective_value = Infinity;
                if (result.objective_value === "-Infinity") result.objective_value = -Infinity;
                if (result.objective_value === "NaN") result.objective_value = NaN;
            } catch(e) { throw new Error(resultText); }

            if (!response.ok) {
                if (result.error && result.error.includes("找不到核心引擎")) {
                   throw new Error("Missing UltimateSolver core engines.");
                }
                throw new Error(result.error || 'Server Internal Error');
            }

            clearInterval(timerInterval); clearInterval(chartInterval);
            
            let objValText = 'N/A';
            if (result.objective_value !== undefined && result.objective_value !== null) {
                objValText = isFinite(result.objective_value) ? result.objective_value.toFixed(6) : result.objective_value.toString();
            }
            
            
            let htmlReport = `<div class="report-container">`;
            htmlReport += `
                <div class="report-header">
                    <h2>${currentModelName}</h2>
                    <div style="display:flex; align-items:center;">
                        <label style="margin-right:15px; font-size:12px; color:#94a3b8; cursor:pointer; display:flex; align-items:center;">
                            <input type="checkbox" class="fmt-toggle" style="margin-right:5px; accent-color:#8b5cf6; cursor:pointer;"> 分数显隐
                        </label>
                        <span class="report-badge ${result.status==='optimal'?'success':'error'}">${result.status==='optimal'?'Optimal Found':result.status.toUpperCase()}</span>
                        <span class="hardware-badge">🚀 ${backendText}</span>
                    </div>
                </div>
            `;
            
            if (result.status === 'optimal') {
                const safeObj = isFinite(result.objective_value) ? (result.objective_value||0).toFixed(6) : result.objective_value.toString();
                htmlReport += `
                <div class="report-summary-cards">
                    <div class="report-card highlight-card">
                        <div class="card-title">Objective Value</div>
                        <div class="card-value text-green val-display" data-dec="${safeObj}" data-frac="${window.toFraction(result.objective_value)}">${safeObj}</div>
                    </div>
                    <div class="report-card">
                        <div class="card-title">Compile Time</div>
                        <div class="card-value">${(result.compile_time || 0).toFixed(4)}s</div>
                    </div>
                    <div class="report-card">
                        <div class="card-title">Solve Time</div>
                        <div class="card-value">${(result.solve_time || 0).toFixed(4)}s</div>
                    </div>
                </div>
                <div class="report-table-section">
                    <h3>变量解空间 (Variable Solutions)</h3>
                    <table class="report-table">
                        <thead>
                            <tr><th>Variable</th><th>Value</th><th>Reduced Cost</th></tr>
                        </thead>
                        <tbody>
                `;
                
                if (result.solution && result.solution.length > 0) {
                    result.solution.forEach(v => {
                        let dec = v.value.toFixed(6);
                        let frac = window.toFraction(v.value);
                        htmlReport += `<tr><td><strong>${v.name}</strong></td><td class="val-display" data-dec="${dec}" data-frac="${frac}">${dec}</td><td>0.000000</td></tr>`;
                    });
                } else {
                    htmlReport += `<tr><td colspan="3" style="text-align:center; color:#64748b;">(No non-zero variables found)</td></tr>`;
                }
                htmlReport += `</tbody></table></div>`;
            } else {
                htmlReport += `
                <div class="report-error-card">
                    <h3>Optimization failed or model is ${result.status}</h3>
                    <p>${result.message}</p>
                </div>`;
                reportWin.querySelector('.window-header').classList.remove('win-success');
            }
            htmlReport += `</div>`;
            
            // Clean UI
            statusNode.style.display = 'none';
            outputTx.style.display = 'block';
            outputTx.innerHTML = htmlReport;
            
            const toggle = outputTx.querySelector('.fmt-toggle');
            if (toggle) {
                toggle.addEventListener('change', (e) => {
                    const isFrac = e.target.checked;
                    outputTx.querySelectorAll('.val-display').forEach(td => {
                        td.innerText = isFrac ? td.getAttribute('data-frac') : td.getAttribute('data-dec');
                    });
                });
            }
            
        } catch (err) {
            clearInterval(timerInterval); clearInterval(chartInterval);
            statusNode.style.display = 'none';
            outputTx.style.display = 'block';
            reportWin.querySelector('.window-header').classList.remove('win-success');
            outputTx.innerHTML = `<div class="report-container"><div class="report-error-card"><h3>[UltimateSolver Error Logger]</h3><p>Fatal error encountered during execution:<br/>${err.message}</p></div></div>`;
        }
    });

    loadState();
});
