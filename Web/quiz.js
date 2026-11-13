(function () {
  'use strict';
  const STORAGE_KEY = 'ml-textbook-progress-kk-v1';
  const emptyChapter = () => ({read: false, answers: {}});
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  function normalizeChapter(value) {
    const result = emptyChapter();
    if (!value || typeof value !== 'object') return result;
    result.read = value.read === true;
    if (value.answers && typeof value.answers === 'object') {
      Object.entries(value.answers).forEach(([key, choice]) => {
        if (/^[a-zA-Z0-9_]+$/.test(key) && Number.isInteger(choice) && choice >= 0 && choice < 4) {
          result.answers[key] = choice;
        }
      });
    }
    return result;
  }

  function createStore(storage) {
    let persistent = Boolean(storage);
    let state = {version: 1, chapters: {}};
    if (persistent) {
      try {
        storage.setItem(STORAGE_KEY + '-check', '1');
        storage.removeItem(STORAGE_KEY + '-check');
        const saved = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
        if (saved && saved.version === 1 && saved.chapters && typeof saved.chapters === 'object') {
          Object.entries(saved.chapters).forEach(([id, value]) => {
            if (/^ch\d{2}$/.test(id)) state.chapters[id] = normalizeChapter(value);
          });
        }
      } catch (_) {
        persistent = false;
      }
    }
    function save() {
      if (persistent) {
        try { storage.setItem(STORAGE_KEY, JSON.stringify(state)); }
        catch (_) { persistent = false; }
      }
    }
    return {
      get persistent() { return persistent; },
      chapter(id) { return normalizeChapter(state.chapters[id]); },
      updateChapter(id, updater) {
        state.chapters[id] = normalizeChapter(updater(normalizeChapter(state.chapters[id])));
        save();
        return this.chapter(id);
      },
      resetChapter(id) { state.chapters[id] = emptyChapter(); save(); }
    };
  }

  function summarize(questions, answers) {
    const result = {answered: 0, correct: 0, total: questions.length, complete: false};
    questions.forEach(question => {
      const answer = answers && answers[question.id];
      if (Number.isInteger(answer) && answer >= 0 && answer < question.options.length) {
        result.answered++;
        if (answer === question.correct) result.correct++;
      }
    });
    result.complete = result.total > 0 && result.answered === result.total;
    return result;
  }

  function summarizeCourse(chapters, getChapter) {
    const rows = chapters.map(meta => {
      const state = normalizeChapter(getChapter(meta.id));
      const ids = Array.isArray(meta.questionIds) ? meta.questionIds : [];
      const answered = ids.filter(id => Number.isInteger(state.answers[id]) && state.answers[id] >= 0 && state.answers[id] < 4).length;
      return {id: meta.id, read: state.read, answered, total: ids.length, complete: state.read && ids.length > 0 && answered === ids.length};
    });
    return {
      chapters: rows,
      chaptersAvailable: rows.length,
      chaptersCompleted: rows.filter(row => row.complete).length,
      answered: rows.reduce((sum, row) => sum + row.answered, 0),
      total: rows.reduce((sum, row) => sum + row.total, 0)
    };
  }

  const api = {createStore, summarize, summarizeCourse};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  let browserStorage;
  try { browserStorage = window.localStorage; } catch (_) { browserStorage = null; }
  const store = createStore(browserStorage);
  const chapterId = document.body.dataset.chapter || 'ch01';
  const quizData = document.getElementById('quiz-data');
  const questions = quizData ? JSON.parse(quizData.textContent) : [];
  const courseData = document.getElementById('course-data');
  const courseChapters = courseData ? JSON.parse(courseData.textContent) : [{id: chapterId, questionIds: questions.map(question => question.id)}];
  const quizRoot = document.getElementById('quiz-questions');
  const letters = ['A', 'B', 'C', 'D'];
  const statusRegion = document.getElementById('action-status');

  function announce(message) {
    if (statusRegion) statusRegion.textContent = message;
  }

  function updateProgress() {
    const chapter = store.chapter(chapterId);
    const result = questions.length ? summarize(questions, chapter.answers) : null;
    const answered = result ? result.answered : Object.keys(chapter.answers).length;
    const total = result ? result.total : 15;
    document.querySelectorAll('[data-answered]').forEach(el => { el.textContent = String(answered); });
    document.querySelectorAll('[data-quiz-total]').forEach(el => { el.textContent = String(total); });
    document.querySelectorAll('[data-progress]').forEach(el => { el.max = total; el.value = answered; });
    document.querySelectorAll('[data-read-status]').forEach(el => {
      el.textContent = chapter.read ? 'Оқылғаны белгіленді' : 'Оқылғаны белгіленбеді';
    });
    const courseProgress = summarizeCourse(courseChapters, id => store.chapter(id));
    document.querySelectorAll('[data-chapter-status]').forEach(el => {
      const targetId = el.dataset.chapterStatus || chapterId;
      const row = courseProgress.chapters.find(item => item.id === targetId);
      const complete = row ? row.complete : chapter.read && answered === total;
      const count = row ? row.answered : answered;
      el.textContent = complete ? 'Тарау аяқталды' : count ? 'Оқу жалғасуда' : 'Қолжетімді';
    });
    document.querySelectorAll('[data-course-available]').forEach(el => { el.textContent = String(courseProgress.chaptersAvailable); });
    document.querySelectorAll('[data-course-completed]').forEach(el => { el.textContent = String(courseProgress.chaptersCompleted); });
    document.querySelectorAll('[data-course-answered]').forEach(el => { el.textContent = String(courseProgress.answered); });
    document.querySelectorAll('[data-course-total]').forEach(el => { el.textContent = String(courseProgress.total); });
    document.querySelectorAll('[data-course-progress]').forEach(el => { el.max = courseProgress.total || 1; el.value = courseProgress.answered; });
    document.querySelectorAll('[data-storage-status]').forEach(el => {
      el.textContent = store.persistent
        ? 'Оқу барысы осы браузерде сақталады. Жергілікті файлдарда тараулар арасындағы сақтау браузер мүмкіндігіне байланысты.'
        : 'Браузер жергілікті сақтауды шектеді. Тест жұмыс істейді, оқу барысы бет жабылғанға дейін қолжетімді.';
    });
    const readBox = document.getElementById('mark-read');
    if (readBox) readBox.checked = chapter.read;
    const resultText = document.getElementById('quiz-result');
    if (resultText && result) {
      resultText.textContent = result.complete
        ? `Тест аяқталды. ${result.total} сұрақтың ${result.correct} жауабы дұрыс (${Math.round(result.correct / result.total * 100)}%). Бұл — өзіндік тексеру. Ресми бақылауды оқытушы тағайындайды.`
        : `Тексерілген жауаптар ${result.answered} / ${result.total}. Дұрыс жауаптар ${result.correct}.`;
    }
  }

  function renderQuiz() {
    if (!quizRoot) return;
    quizRoot.replaceChildren();
    const answers = store.chapter(chapterId).answers;
    questions.forEach((question, index) => {
      const form = document.createElement('form');
      form.className = 'quiz-question';
      const fieldset = document.createElement('fieldset');
      const legend = document.createElement('legend');
      legend.textContent = `Сұрақ ${index + 1} / ${questions.length}`;
      const text = document.createElement('p');
      text.id = question.id + '-text';
      text.className = 'question-text';
      text.textContent = question.question;
      fieldset.setAttribute('aria-describedby', text.id);
      fieldset.append(legend, text);
      const labels = [];
      question.options.forEach((option, choice) => {
        const label = document.createElement('label');
        label.className = 'quiz-option';
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = question.id;
        input.value = String(choice);
        input.required = true;
        input.id = question.id + '-' + choice;
        const span = document.createElement('span');
        span.textContent = `${letters[choice]}. ${option}`;
        label.append(input, span);
        labels.push(label);
        fieldset.append(label);
      });
      const button = document.createElement('button');
      button.type = 'submit';
      button.className = 'button';
      button.textContent = 'Жауапты тексеру';
      const feedback = document.createElement('div');
      feedback.className = 'quiz-feedback';
      feedback.setAttribute('role', 'status');
      feedback.setAttribute('aria-live', 'polite');
      feedback.hidden = true;
      form.append(fieldset, button, feedback);

      function showSubmitted(selected) {
        const isCorrect = selected === question.correct;
        fieldset.disabled = true;
        fieldset.querySelector(`input[value="${selected}"]`).checked = true;
        labels[selected].classList.add('selected-answer');
        const outcome = document.createElement('p');
        outcome.className = 'answer-outcome';
        outcome.textContent = isCorrect
          ? 'Жауап дұрыс.'
          : `Жауап қате. Дұрыс нұсқа ${letters[question.correct]}. ${question.options[question.correct]}`;
        const explanation = document.createElement('p');
        explanation.textContent = question.explanation;
        feedback.replaceChildren(outcome, explanation);
        feedback.hidden = false;
        feedback.classList.toggle('answer-correct', isCorrect);
        button.hidden = true;
      }

      if (own(answers, question.id)) showSubmitted(answers[question.id]);
      form.addEventListener('submit', event => {
        event.preventDefault();
        const chosen = form.querySelector('input:checked');
        if (!chosen) { form.reportValidity(); return; }
        const selected = Number(chosen.value);
        store.updateChapter(chapterId, current => ({...current, answers: {...current.answers, [question.id]: selected}}));
        showSubmitted(selected);
        updateProgress();
      });
      quizRoot.append(form);
    });
  }

  function buildContents() {
    const article = document.getElementById('chapter-content');
    const list = document.getElementById('chapter-contents');
    if (!article || !list) return;
    article.querySelectorAll('h2, h3').forEach((heading, index) => {
      if (!heading.id) heading.id = `section-${index + 1}`;
      const item = document.createElement('li');
      if (heading.tagName === 'H3') item.className = 'contents-subsection';
      const link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent;
      item.append(link);
      list.append(item);
    });
  }

  const markRead = document.getElementById('mark-read');
  if (markRead) markRead.addEventListener('change', () => {
    store.updateChapter(chapterId, chapter => ({...chapter, read: markRead.checked}));
    updateProgress();
    announce(markRead.checked ? 'Тараудың оқылғаны белгіленді.' : 'Оқылғаны туралы белгі алынды.');
  });
  document.querySelectorAll('[data-reset-progress]').forEach(button => {
    button.addEventListener('click', () => {
      store.resetChapter(chapterId);
      renderQuiz();
      updateProgress();
      const panel = button.closest('details');
      if (panel) panel.open = false;
      announce('Жауаптар мен оқу белгісі тазартылды. Тестті қайта орындауға болады.');
      const firstInput = quizRoot && quizRoot.querySelector('input');
      if (firstInput) firstInput.focus();
    });
  });
  renderQuiz();
  buildContents();
  updateProgress();
})();
