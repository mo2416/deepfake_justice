const input = document.querySelector('#fileInput');
const empty = document.querySelector('#emptyState');
const state = document.querySelector('#previewState');
const preview = document.querySelector('#preview');
const button = document.querySelector('#scanButton');
const status = document.querySelector('#scanStatus');
const note = document.querySelector('#demoNote');

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  empty.hidden = true;
  state.hidden = false;
  button.disabled = false;
  status.textContent = `Ready · ${file.name}`;
  note.hidden = true;
});

button.addEventListener('click', (event) => {
  event.preventDefault();
  if (!input.files?.length) return;
  button.disabled = true;
  state.classList.remove('scanning');
  void state.offsetWidth;
  state.classList.add('scanning');
  status.textContent = 'Scanning interface preview…';
  note.hidden = true;
  setTimeout(() => {
    state.classList.remove('scanning');
    status.textContent = 'Interface preview complete';
    note.hidden = false;
    button.disabled = false;
  }, 4500);
});
