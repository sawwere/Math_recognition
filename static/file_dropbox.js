const dropZone = document.getElementById("file_dropbox");
if (dropZone) {
    let hoverClassName = 'hover';

    dropZone.addEventListener("dragenter", function (e) {
        e.preventDefault();
        dropZone.classList.add(hoverClassName);
    });

    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add(hoverClassName);
    });

    dropZone.addEventListener("dragleave", function (e) {
        e.preventDefault();
        dropZone.classList.remove(hoverClassName);
    });

    // Это самое важное событие, событие, которое дает доступ к файлам
    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove(hoverClassName);

        const files = Array.from(e.dataTransfer.files);
        console.log(files);
        let fileInputElement = document.getElementById('file_input');
        let container = new DataTransfer();
        container.items.add(files[0]);
        fileInputElement.files = container.files;
    });
}