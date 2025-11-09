document.addEventListener("DOMContentLoaded", () => {
    const counters = document.querySelectorAll(".stat");
    const options = { threshold: 0.4 };

    const animate = (entry) => {
        const target = entry.target;
        const value = parseInt(target.dataset.target, 10);
        const span = target.querySelector(".stat-value");
        let count = 0;
        const step = Math.ceil(value / 80);
        const ticker = () => {
            count += step;
            if (count >= value) {
                span.textContent = value + "+";
                return;
            }
            span.textContent = count;
            requestAnimationFrame(ticker);
        };
        requestAnimationFrame(ticker);
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                animate(entry);
                observer.unobserve(entry.target);
            }
        });
    }, options);

    counters.forEach((stat) => observer.observe(stat));
});
