document.getElementById('openModalBtn').addEventListener('click', function () {
  document.getElementById('modal').style.display = 'block';
});

document.getElementById('closeModalBtn').addEventListener('click', function () {
  document.getElementById('modal').style.display = 'none';
});

// Close modal if the user clicks outside the modal content
window.addEventListener('click', function (event) {
  var modal = document.getElementById('modal');
  if (event.target === modal) {
      modal.style.display = 'none';
  }
});



