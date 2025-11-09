(function () {
  const pipelineRoot = document.documentElement.classList.contains('nlp-pipeline')
    ? document.documentElement
    : null;

  if (!pipelineRoot) {
    return;
  }

  const jobId = document.documentElement.dataset.jobId;
  const endpoints = {
    upload: '/nlp/upload',
    runStep: '/nlp/run_step',
    artifacts: (id) => `/nlp/artifacts/${encodeURIComponent(id)}`,
    downloadArtifact: (id, filename) => `/nlp/artifacts/${encodeURIComponent(id)}/download/${encodeURIComponent(filename)}`
  };

  const dom = {
    currentStep: document.getElementById('nlp-current-step'),
    lastAction: document.getElementById('nlp-last-action'),
    dataQuality: document.getElementById('nlp-data-quality'),
    artifactCount: document.getElementById('nlp-artifact-count'),
    ingestionPreview: document.getElementById('ingestion-preview'),
    ingestionSummary: document.getElementById('ingestion-summary'),
    profilingStats: document.getElementById('profiling-stats'),
    profilingUnigrams: document.getElementById('profiling-unigrams'),
    profilingBigrams: document.getElementById('profiling-bigrams'),
    profilingLength: document.getElementById('profiling-length'),
    cleaningExamples: document.getElementById('cleaning-examples'),
    cleaningSummary: document.getElementById('cleaning-summary'),
    cleaningDownloads: document.getElementById('cleaning-downloads'),
    featuresSummary: document.getElementById('features-summary'),
    featuresTopTerms: document.getElementById('features-top-terms'),
    featuresDownloads: document.getElementById('features-downloads'),
    trainMetrics: document.getElementById('train-metrics'),
    trainConfusion: document.getElementById('train-confusion'),
    trainMisclassified: document.getElementById('train-misclassified'),
    trainDownloads: document.getElementById('train-downloads'),
    artifactList: document.getElementById('artifact-list'),
    auditList: document.getElementById('audit-list')
  };

  const formSelectors = {
    textColumn: document.querySelectorAll('[data-role="text-column-select"]'),
    labelColumn: document.querySelectorAll('[data-role="label-column-select"]'),
    mlpOptions: document.querySelector('[data-mlp-options]')
  };

  function toggleSpinner(step, on) {
    const spinner = document.querySelector(`[data-spinner="${step}"]`);
    if (spinner) {
      spinner.hidden = !on;
    }
    const button = document.querySelector(`[data-run-step="${step}"]`);
    if (button) {
      button.disabled = on;
    }
  }

  function updateSelectOptions(nodeList, values, selected) {
    nodeList.forEach((select) => {
      const currentValue = selected || select.value;
      select.innerHTML = '';
      if (select.dataset.role === 'label-column-select') {
        select.insertAdjacentHTML('beforeend', '<option value="">None</option>');
      } else {
        select.insertAdjacentHTML('beforeend', '<option value="">Auto-detect</option>');
      }
      values.forEach((value) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        if (value === currentValue) {
          option.selected = true;
        }
        select.appendChild(option);
      });
    });
  }

  function renderTable(tableEl, rows) {
    if (!tableEl) return;
    if (!rows || !rows.length) {
      tableEl.innerHTML = '<caption class="muted">No data available</caption>';
      return;
    }
    const columns = Object.keys(rows[0]);
    const thead = `<thead><tr>${columns.map((c) => `<th>${c}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${rows
      .map((row) => `<tr>${columns.map((c) => `<td>${row[c]}</td>`).join('')}</tr>`)
      .join('')}</tbody>`;
    tableEl.innerHTML = `${thead}${tbody}`;
  }

  function renderList(target, items) {
    if (!target) return;
    if (!items || !items.length) {
      target.innerHTML = '<p class="muted">No data available.</p>';
      return;
    }
    target.innerHTML = items
      .map(
        (item) => `<div class="artifact-card">
            <header>
                <span>${item.name}</span>
                <a class="ghost" href="${item.url}">Download</a>
            </header>
            <div class="artifact-meta">
                <span>${item.size}</span>
                <span>${item.ts}</span>
            </div>
        </div>`
      )
      .join('');
  }

  function updateHero(job) {
    if (!job) return;
    if (dom.currentStep) dom.currentStep.textContent = job.current_step ? capitalize(job.current_step) : '—';
    if (dom.lastAction) dom.lastAction.textContent = job.last_action || 'n/a';
    if (dom.dataQuality && job.data_quality) {
      if (job.data_quality.raw) {
        dom.dataQuality.textContent = `Raw ${job.data_quality.raw} → Cleaned ${job.data_quality.cleaned ?? '?'}`;
      } else {
        dom.dataQuality.textContent = 'Not scored';
      }
    }
    if (dom.artifactCount) dom.artifactCount.textContent = job.artifact_count ?? 0;
    highlightSteps(job);
  }

  function highlightSteps(job) {
    const completed = new Set(job.completed_steps || []);
    document.querySelectorAll('[data-step-link]').forEach((link) => {
      const key = link.dataset.stepLink;
      link.classList.toggle('is-active', key === job.current_step);
      link.classList.toggle('is-complete', completed.has(key));
    });
  }

  function renderAudit(audit) {
    if (!dom.auditList) return;
    if (!audit || !audit.length) {
      dom.auditList.innerHTML = '<li><p class="muted">No audit entries yet.</p></li>';
      return;
    }
    dom.auditList.innerHTML = audit
      .map(
        (entry) => `<li>
          <span class="audit-time">${entry.timestamp}</span>
          <strong>${capitalize(entry.step || '')}</strong>
          <p>${entry.summary}</p>
        </li>`
      )
      .join('');
  }

  function renderProfiling(result) {
    if (!result) return;
    if (dom.profilingStats) {
      dom.profilingStats.innerHTML = [
        { label: 'Documents', value: result.n_docs },
        { label: 'Empty entries', value: result.n_empty },
        { label: 'Avg tokens', value: result.avg_tokens?.toFixed(1) },
        { label: 'Median tokens', value: result.median_tokens },
        { label: 'Vocabulary size', value: result.vocab_size }
      ]
        .map(
          (item) => `<article class="stat-card">
            <span class="stat-label">${item.label}</span>
            <span class="stat-value">${item.value ?? '—'}</span>
        </article>`
        )
        .join('');
    }
    renderTable(
      dom.profilingUnigrams,
      (result.top_unigrams || []).map(([term, count]) => ({ term, count }))
    );
    renderTable(
      dom.profilingBigrams,
      (result.top_bigrams || []).map(([term, count]) => ({ term, count }))
    );
    if (dom.profilingLength) {
      dom.profilingLength.innerHTML = `<ul>${(result.length_distribution || [])
        .map((bucket) => `<li>${bucket.bucket}: ${bucket.count}</li>`)
        .join('')}</ul>`;
    }
  }

  function renderCleaning(result) {
    if (dom.cleaningExamples) {
      if (result.examples && result.examples.length) {
        dom.cleaningExamples.innerHTML = result.examples
          .map(
            (example) => `<article>
            <h3>Original</h3>
            <p>${example.original || '<em>empty</em>'}</p>
            <h3>Cleaned</h3>
            <p>${example.cleaned || '<em>empty</em>'}</p>
        </article>`
          )
          .join('');
      } else {
        dom.cleaningExamples.innerHTML = '<p class="muted">Preview pending.</p>';
      }
    }
    if (dom.cleaningSummary) {
      dom.cleaningSummary.textContent = result.summary || '';
    }
    if (dom.cleaningDownloads) {
      dom.cleaningDownloads.innerHTML = (result.artifacts || [])
        .map(
          (artifact) => `<a class="ghost" href="${artifact.url}">
            ${artifact.name} (${artifact.size})
        </a>`
        )
        .join('');
    }
  }

  function renderFeatures(result) {
    if (dom.featuresSummary) {
      dom.featuresSummary.textContent = result.summary || '';
    }
    renderTable(dom.featuresTopTerms, result.top_terms || []);
    if (dom.featuresDownloads) {
      dom.featuresDownloads.innerHTML = (result.artifacts || [])
        .map((artifact) => `<a class="ghost" href="${artifact.url}">${artifact.name} (${artifact.size})</a>`)
        .join('');
    }
  }

  function renderTraining(result) {
    if (dom.trainMetrics) {
      dom.trainMetrics.innerHTML = Object.entries(result.metrics || {})
        .map(
          ([name, value]) => `<article class="stat-card">
            <span class="stat-label">${capitalize(name)}</span>
            <span class="stat-value">${typeof value === 'number' ? value.toFixed(3) : value}</span>
        </article>`
        )
        .join('');
    }
    if (dom.trainConfusion) {
      const labels = result.confusion_labels || [];
      const matrix = result.confusion || [];
      if (!matrix.length) {
        dom.trainConfusion.innerHTML = '<p class="muted">Confusion matrix pending.</p>';
      } else {
        const headers = `<tr><th>Actual \\ Pred</th>${labels.map((l) => `<th>${l}</th>`).join('')}</tr>`;
        const body = matrix
          .map(
            (row, idx) =>
              `<tr><td>${labels[idx]}</td>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`
          )
          .join('');
        dom.trainConfusion.innerHTML = `<table class="data-table">${headers}${body}</table>`;
      }
    }
    if (dom.trainMisclassified) {
      const samples = result.misclassified_samples || [];
      if (!samples.length) {
        dom.trainMisclassified.innerHTML = '<p class="muted">No misclassified samples captured.</p>';
      } else {
        dom.trainMisclassified.innerHTML = `<h3>Misclassified examples</h3>${samples
          .map(
            (s) => `<article class="artifact-card">
              <p>${s.text}</p>
              <div class="artifact-meta">
                <span>True: ${s.true}</span>
                <span>Pred: ${s.pred}</span>
              </div>
          </article>`
          )
          .join('')}`;
      }
    }
    if (dom.trainDownloads) {
      dom.trainDownloads.innerHTML = (result.artifacts || [])
        .map((artifact) => `<a class="ghost" href="${artifact.url}">${artifact.name}</a>`)
        .join('');
    }
  }

  function renderArtifacts(artifacts) {
    renderList(dom.artifactList, artifacts);
  }

  function fetchArtifacts() {
    fetch(endpoints.artifacts(jobId))
      .then((res) => res.json())
      .then((payload) => {
        renderArtifacts(payload.artifacts || []);
      })
      .catch(() => {
        renderArtifacts([]);
      });
  }

  function capitalize(value) {
    if (!value) return '';
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  function handleUpload(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    formData.set('job_id', jobId);
    const spinner = form.querySelector('.spinner');
    if (spinner) spinner.hidden = false;
    fetch(endpoints.upload, {
      method: 'POST',
      body: formData
    })
      .then((response) => response.json())
      .then((payload) => {
        if (payload.error) throw new Error(payload.error);
        renderTable(dom.ingestionPreview, payload.preview);
        renderSummary(dom.ingestionSummary, payload.summary);
        updateSelectOptions(formSelectors.textColumn, payload.columns || [], payload.text_column);
        updateSelectOptions(formSelectors.labelColumn, payload.columns || [], payload.label_column);
        updateHero(payload.job);
        renderAudit(payload.audit);
        if (payload.job && payload.job.current_step === 'profiling') {
          runStep('profiling', {});
        }
      })
      .catch((error) => showToast(error.message || 'Upload failed'))
      .finally(() => {
        if (spinner) spinner.hidden = true;
      });
  }

  function renderSummary(container, summary) {
    if (!container) return;
    if (!summary) {
      container.innerHTML = '';
      return;
    }
    container.innerHTML = Object.entries(summary)
      .map(
        ([label, value]) => `<div>
          <dt>${label}</dt>
          <dd>${value}</dd>
        </div>`
      )
      .join('');
  }

  function runStep(step, options) {
    toggleSpinner(step, true);
    fetch(endpoints.runStep, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: jobId,
        step,
        options
      })
    })
      .then((response) => response.json())
      .then((payload) => {
        if (payload.error) throw new Error(payload.error);
        updateHero(payload.job);
        renderAudit(payload.audit);
        switch (step) {
          case 'profiling':
            renderProfiling(payload.result);
            break;
          case 'cleaning':
            renderCleaning(payload.result);
            break;
          case 'features':
            renderFeatures(payload.result);
            break;
          case 'train':
            renderTraining(payload.result);
            break;
          case 'reports':
            renderArtifacts(payload.artifacts);
            break;
          default:
            break;
        }
        if (payload.artifacts) {
          renderArtifacts(payload.artifacts);
        }
      })
      .catch((error) => showToast(error.message || `Step ${step} failed`))
      .finally(() => {
        toggleSpinner(step, false);
      });
  }

  function gatherFormOptions(form) {
    const data = new FormData(form);
    const options = {};
    data.forEach((value, key) => {
      if (value === '' || key === 'job_id') return;
      if (value === 'on') {
        options[key] = true;
      } else if (!Number.isNaN(Number(value)) && value !== '') {
        options[key] = Number(value);
      } else {
        options[key] = value;
      }
    });
    form.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
      if (!checkbox.name) return;
      options[checkbox.name] = checkbox.checked;
    });
    return options;
  }

  function showToast(message) {
    if (!message) return;
    console.warn(message);
    const toast = document.createElement('div');
    toast.className = 'flash flash--danger';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }

  function initNavigation() {
    document.querySelectorAll('[data-step-link]').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  function initForms() {
    const uploadForm = document.getElementById('nlp-upload-form');
    if (uploadForm) {
      uploadForm.addEventListener('submit', handleUpload);
    }
    const profilingBtn = document.querySelector('[data-run-step="profiling"]');
    if (profilingBtn) {
      profilingBtn.addEventListener('click', () => runStep('profiling', {}));
    }
    const cleaningForm = document.getElementById('cleaning-form');
    if (cleaningForm) {
      cleaningForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const options = gatherFormOptions(cleaningForm);
        runStep('cleaning', options);
      });
      const previewBtn = cleaningForm.querySelector('[data-preview-cleaning]');
      if (previewBtn) {
        previewBtn.addEventListener('click', () => {
          const options = gatherFormOptions(cleaningForm);
          options.preview_only = true;
          runStep('cleaning', options);
        });
      }
    }
    const featuresForm = document.getElementById('features-form');
    if (featuresForm) {
      featuresForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const options = gatherFormOptions(featuresForm);
        runStep('features', options);
      });
    }
    const trainForm = document.getElementById('train-form');
    if (trainForm) {
      trainForm.addEventListener('change', (event) => {
        if (event.target.name === 'model_type' && formSelectors.mlpOptions) {
          formSelectors.mlpOptions.hidden = event.target.value !== 'mlp';
        }
      });
      trainForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const options = gatherFormOptions(trainForm);
        if (options.test_size) {
          options.test_size = Number(options.test_size) / 100;
        }
        runStep('train', options);
      });
    }
    const reportBtn = document.querySelector('[data-generate-report]');
    if (reportBtn) {
      reportBtn.addEventListener('click', () => runStep('reports', {}));
    }
  }

  initNavigation();
  initForms();
  fetchArtifacts();
})();
