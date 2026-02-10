# 🎨 Guide de Personnalisation de l'Interface Web

Ce guide vous aide à personnaliser l'interface web selon vos besoins.

## 🎨 Personnaliser les Couleurs

### Modifier les couleurs principales

Éditez `css/style.css` et changez les variables CSS :

```css
:root {
    --primary: #6366f1;        /* Couleur principale */
    --secondary: #8b5cf6;      /* Couleur secondaire */
    --success: #10b981;        /* Vert (succès) */
    --warning: #f59e0b;        /* Orange (avertissement) */
    --danger: #ef4444;         /* Rouge (erreur) */
}
```

### Exemples de palettes

**Palette Bleue (Ocean)** :
```css
--primary: #0ea5e9;
--secondary: #0284c7;
```

**Palette Verte (Nature)** :
```css
--primary: #10b981;
--secondary: #059669;
```

**Palette Rose (Sunset)** :
```css
--primary: #ec4899;
--secondary: #db2777;
```

**Palette Orange (Fire)** :
```css
--primary: #f97316;
--secondary: #ea580c;
```

## 🖼️ Changer le Fond d'Écran

### Gradient actuel
```css
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### Alternatives

**Gradient Bleu-Vert** :
```css
background: linear-gradient(135deg, #667eea 0%, #06b6d4 100%);
```

**Gradient Rose-Orange** :
```css
background: linear-gradient(135deg, #ec4899 0%, #f97316 100%);
```

**Gradient Sombre** :
```css
background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
```

**Image de fond** :
```css
background: url('assets/background.jpg') no-repeat center center fixed;
background-size: cover;
```

## ✨ Modifier les Animations

### Vitesse des transitions

```css
/* Plus rapide */
.tp-card {
    transition: all 0.15s;
}

/* Plus lente */
.tp-card {
    transition: all 0.6s;
}
```

### Désactiver les animations

```css
* {
    transition: none !important;
    animation: none !important;
}
```

## 🎯 Ajouter un Nouveau TP

### 1. Ajouter la configuration

Dans `js/config.js` :

```javascript
TP_CONFIGS.mon_nouveau_tp = {
    title: 'TP6 - Mon Nouveau TP',
    description: 'Description de mon TP',
    icon: 'fa-star',
    steps: [
        { id: 'step1', label: 'Étape 1', completed: false },
        { id: 'step2', label: 'Étape 2', completed: false }
    ]
};
```

### 2. Ajouter la carte dans index.html

```html
<div class="tp-card" onclick="loadTP('mon_nouveau_tp')">
    <div class="tp-icon">
        <i class="fas fa-star"></i>
    </div>
    <h3>TP6 - Mon Nouveau TP</h3>
    <p>Description courte</p>
    <div class="tp-tags">
        <span class="tag">Tag1</span>
        <span class="tag">Tag2</span>
    </div>
</div>
```

### 3. Ajouter le formulaire dans ui.js

```javascript
const forms = {
    // ... autres TPs
    mon_nouveau_tp: `
        <div class="form-group">
            <button class="btn" onclick="handleMonAction()">
                <i class="fas fa-play"></i> Lancer
            </button>
        </div>
    `
};
```

### 4. Ajouter le handler dans app.js

```javascript
async function handleMonAction() {
    UI.showLoading(true);
    try {
        // Votre logique ici
        const result = await API.monEndpoint(STATE.datasetId);
        UI.displayResults('Mon Résultat', result);
    } catch (error) {
        UI.showToast('Erreur', 'error');
    }
    UI.showLoading(false);
}
```

## 🔧 Modifier l'URL de l'API

Dans `js/config.js` :

```javascript
const CONFIG = {
    API_URL: 'http://localhost:8000',  // Local
    // API_URL: 'https://mon-api.com',  // Production
    DEFAULT_SEED: 42,
    DEFAULT_N_ROWS: 1000
};
```

## 📱 Ajuster le Responsive

### Points de rupture

```css
/* Tablettes */
@media (max-width: 1024px) {
    .tp-cards {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Mobiles */
@media (max-width: 768px) {
    .tp-cards {
        grid-template-columns: 1fr;
    }
    .hero-title {
        font-size: 2rem;
    }
}

/* Petits mobiles */
@media (max-width: 480px) {
    .container {
        padding: 0 1rem;
    }
}
```

## 🎭 Thème Sombre

Ajoutez un bouton de toggle et ces styles :

```css
body.dark-theme {
    --dark: #f8fafc;
    --light: #1e293b;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
}

body.dark-theme .tp-card {
    background: #1e293b;
    color: white;
}
```

JavaScript pour le toggle :

```javascript
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    localStorage.setItem('theme', 
        document.body.classList.contains('dark-theme') ? 'dark' : 'light'
    );
}

// Au chargement
if (localStorage.getItem('theme') === 'dark') {
    document.body.classList.add('dark-theme');
}
```

## 🔔 Personnaliser les Notifications

### Position

```css
.toast {
    /* En haut à droite */
    top: 2rem;
    right: 2rem;
    bottom: auto;
    
    /* En haut au centre */
    top: 2rem;
    left: 50%;
    transform: translateX(-50%);
    right: auto;
}
```

### Durée

Dans `js/ui.js` :

```javascript
showToast(message, type = 'info') {
    // ...
    setTimeout(() => {
        toast.classList.remove('show');
    }, 5000); // 5 secondes au lieu de 3
}
```

## 📊 Personnaliser les Graphiques Plotly

### Thème global

```javascript
const defaultLayout = {
    paper_bgcolor: '#f8fafc',
    plot_bgcolor: '#f8fafc',
    font: {
        family: 'Inter, sans-serif',
        size: 12,
        color: '#1e293b'
    }
};

Plotly.newPlot('plotlyChart', data, {...layout, ...defaultLayout});
```

### Couleurs personnalisées

```javascript
const trace = {
    // ...
    marker: {
        color: '#6366f1',  // Votre couleur
        line: {
            color: '#4f46e5',
            width: 1
        }
    }
};
```

## 🖋️ Changer la Police

### Google Fonts

Ajoutez dans `<head>` de index.html :

```html
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
```

Modifiez dans style.css :

```css
body {
    font-family: 'Poppins', sans-serif;
}
```

### Polices disponibles
- **Poppins** : Moderne et arrondi
- **Montserrat** : Géométrique
- **Roboto** : Classique et lisible
- **Raleway** : Élégant
- **Open Sans** : Neutre

## 🎨 Effets Visuels Avancés

### Glass morphism

```css
.tp-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
```

### Neumorphism

```css
.tp-card {
    background: #e0e5ec;
    box-shadow: 
        9px 9px 16px rgba(163, 177, 198, 0.6),
        -9px -9px 16px rgba(255, 255, 255, 0.5);
}
```

### Neon effect

```css
.btn:hover {
    box-shadow: 
        0 0 5px var(--primary),
        0 0 25px var(--primary),
        0 0 50px var(--primary);
}
```

## 🚀 Optimisations de Performance

### Charger Plotly localement

Téléchargez plotly.min.js et :

```html
<!-- Au lieu de -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<!-- Utilisez -->
<script src="assets/plotly.min.js"></script>
```

### Lazy loading des images

```html
<img src="placeholder.jpg" data-src="real-image.jpg" loading="lazy">
```

### Minifier CSS et JS

```bash
# Installer un minifier
npm install -g minify

# Minifier
minify css/style.css > css/style.min.css
minify js/app.js > js/app.min.js
```

## 📝 Bonnes Pratiques

1. **Commentez vos modifications** :
```css
/* Modification : couleur primaire changée en bleu ocean */
--primary: #0ea5e9;
```

2. **Testez sur plusieurs navigateurs** :
- Chrome
- Firefox
- Safari
- Edge

3. **Vérifiez le responsive** :
- Mobile (320px)
- Tablette (768px)
- Desktop (1920px)

4. **Validez votre code** :
- HTML : validator.w3.org
- CSS : jigsaw.w3.org/css-validator

## 🎓 Ressources Utiles

### Couleurs
- **Coolors.co** : Générateur de palettes
- **ColorHunt.co** : Palettes prêtes à l'emploi

### Icônes
- **FontAwesome** : fontawesome.com
- **Heroicons** : heroicons.com

### Animations
- **Animate.css** : animate.style
- **AOS** : michalsnik.github.io/aos

### Gradients
- **WebGradients** : webgradients.com
- **UIGradients** : uigradients.com

---

**Amusez-vous à personnaliser ! 🎨**
