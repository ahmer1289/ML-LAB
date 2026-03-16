document.addEventListener('DOMContentLoaded', () => {
    const imageForm = document.getElementById('image-form');
    if (imageForm) {
        imageForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const f = imageForm.querySelector('input[type=file]').files[0];
            const fd = new FormData(); fd.append('image', f);
            const res = await fetch('/api/image_classify', { method: 'POST', body: fd });
            const json = await res.json();
            document.getElementById('result').textContent = JSON.stringify(json, null, 2);
        });
    }

    const genBtn = document.getElementById('gen');
    if (genBtn) {
        genBtn.addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value;
            const res = await fetch('/api/text_generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
            const json = await res.json();
            document.getElementById('gen-out').textContent = json.text;
        });
    }

    const transBtn = document.getElementById('translate');
    if (transBtn) {
        transBtn.addEventListener('click', async () => {
            const text = document.getElementById('translate-in').value;
            const res = await fetch('/api/translate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
            const json = await res.json();
            document.getElementById('translate-out').textContent = json.text;
        });
    }

    const sentForm = document.getElementById('sent-form');
    if (sentForm) {
        sentForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const f = sentForm.querySelector('input[type=file]').files[0];
            const fd = new FormData(); fd.append('audio', f);
            const res = await fetch('/api/sentiment_voice', { method: 'POST', body: fd });
            const json = await res.json();
            document.getElementById('sent-result').textContent = JSON.stringify(json, null, 2);
        });
    }

    const qaForm = document.getElementById('qa-form');
    if (qaForm) {
        qaForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const f = qaForm.querySelector('input[type=file]').files[0];
            const fd = new FormData(); fd.append('audio', f);
            const res = await fetch('/api/qa_voice', { method: 'POST', body: fd });
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                document.getElementById('qa-download').innerHTML = `<a href="${url}" download="answer.wav">Download Answer Audio</a>`;
            } else {
                const txt = await res.text();
                document.getElementById('qa-download').textContent = txt;
            }
        });
    }
});
