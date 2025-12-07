// --- DỮ LIỆU THEME ---
const themes = {
    pixel: {
        name: "Style Dreamy",
        desc: "Input: 784 features",
        bg: "#fffffe", headline: "#1f1235", btn: "#ff6e6c", secondary: "#67568c", tertiary: "#fbdd74"
    },
    sobel: {
        name: "Style Technology",
        desc: "Input: 1568 features",
        bg: "#f0f4f8", headline: "#094067", btn: "#3da9fc", secondary: "#90b4ce", tertiary: "#d8eefe"
    },
    block: {
        name: "Style Minimalist",
        desc: "Input: 196 features",
        bg: "#f8f5f2", headline: "#232323", btn: "#078080", secondary: "#f45d48", tertiary: "#f8f5f2"
    }
};

// --- CHANGE THEME ---
function changeTheme(modelType) {
    const t = themes[modelType];
    const root = document.documentElement;

    root.style.setProperty('--bg-color', t.bg);
    root.style.setProperty('--headline-color', t.headline);
    root.style.setProperty('--btn-color', t.btn);
    root.style.setProperty('--secondary-color', t.secondary);
    root.style.setProperty('--tertiary-color', t.tertiary);

    document.getElementById('styleName').innerText = "Đang sử dụng: " + t.name;
    document.getElementById('modelInfo').innerText = t.desc;

    if(myChart) {
        myChart.data.datasets[0].backgroundColor = t.btn;
        myChart.update();
    }
}

// --- SETUP CHART ---
const ctxChart = document.getElementById('probChart').getContext('2d');
let myChart = new Chart(ctxChart, {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            label: 'Tỉ lệ %',
            data: Array(10).fill(0),
            backgroundColor: '#ff6e6c',
            borderRadius: 6,
            barThickness: 18
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            y: { beginAtZero: true, max: 100, grid: { display: false }, ticks: { display: false } },
            x: { grid: { display: false }, ticks: { font: { family: 'Poppins' } } }
        },
        animation: { duration: 500 }
    }
});

function updateChart(probabilities) {
    const percents = probabilities.map(p => (p * 100).toFixed(1));
    myChart.data.datasets[0].data = percents;
    myChart.update();
}

// --- SETUP DRAWING ---
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

ctx.lineWidth = 22;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#222';

function getMousePos(evt) {
    var rect = canvas.getBoundingClientRect();
    const clientX = evt.clientX || (evt.touches && evt.touches[0].clientX);
    const clientY = evt.clientY || (evt.touches && evt.touches[0].clientY);
    if(clientX) {
        return { x: clientX - rect.left, y: clientY - rect.top };
    }
    return null;
}

canvas.addEventListener('mousedown', start);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stop);
canvas.addEventListener('mouseout', stop);
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); start(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', stop);

function start(e) {
    isDrawing = true;
    ctx.beginPath();
    const pos = getMousePos(e);
    if(pos) ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    if(pos) { ctx.lineTo(pos.x, pos.y); ctx.stroke(); }
}
function stop() { isDrawing = false; }

// --- OTHER FUNCTIONS ---
function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = "?";
    updateChart(Array(10).fill(0));
    ctx.beginPath();
}

function handleImageUpload(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var img = new Image();
            img.onload = function() {
                clearCanvas();
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            }
            img.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
    input.value = '';
}

function predict() {
    const modelType = document.getElementById('modelSelect').value;
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'digit.png');
        formData.append('model_type', modelType);

        document.getElementById('result').innerText = "⏳";

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if(data.error) {
                alert("Lỗi: " + data.error);
                document.getElementById('result').innerText = "!";
            } else {
                document.getElementById('result').innerText = data.digit;
                updateChart(data.probabilities);
            }
        })
        .catch(err => {
            console.error(err);
            alert("Lỗi kết nối Server!");
            document.getElementById('result').innerText = "?";
        });
    });
}

// STARTUP
window.onload = function() {
    clearCanvas();
};
