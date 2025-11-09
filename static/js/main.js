document.addEventListener("DOMContentLoaded", () => {
    const downloadMenus = document.querySelectorAll(".download-menu");

    downloadMenus.forEach((menu) => {
        const button = menu.querySelector(".download-menu__button");
        const list = menu.querySelector(".download-menu__list");

        if (!button || !list) return;

        const toggleMenu = (open) => {
            if (open) {
                list.style.display = "block";
                list.classList.add("is-open");
                button.setAttribute("aria-expanded", "true");
            } else {
                list.style.display = "none";
                list.classList.remove("is-open");
                button.setAttribute("aria-expanded", "false");
            }
        };

        // Toggle on button click
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            const isOpen = list.classList.contains("is-open");
            toggleMenu(!isOpen);
        });

        // Close when clicking outside
        document.addEventListener("click", (event) => {
            if (!menu.contains(event.target)) {
                toggleMenu(false);
            }
        });

        // Close on Escape key
        button.addEventListener("keydown", (event) => {
            if (event.key === "Escape") {
                toggleMenu(false);
                button.focus();
            }
        });

        // Prevent menu from closing when clicking inside the list
        list.addEventListener("click", (event) => {
            event.stopPropagation();
        });
    });

    // Upload dropzone drag and drop enhancement
    const dropzones = document.querySelectorAll("[data-dropzone]");
    dropzones.forEach((zone) => {
        ["dragenter", "dragover"].forEach((eventName) => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.add("is-dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                zone.classList.remove("is-dragover");
            });
        });
    });

    // Auto-dismiss flash messages after 5 seconds
    const flashes = document.querySelectorAll(".flash");
    flashes.forEach((flash) => {
        setTimeout(() => {
            flash.classList.add("flash--dismissed");
            setTimeout(() => flash.remove(), 400);
        }, 5000);
    });
});
