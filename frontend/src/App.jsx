import { useState, useRef } from 'react'

const API = ''
const THETA = 0.906

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const c = {
  card: {
    background: 'var(--bg2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--r-lg)',
    marginBottom: 16,
  },
  cardGlow: {
    background: 'var(--bg2)',
    border: '1px solid var(--bolt-dim)',
    borderRadius: 'var(--r-lg)',
    marginBottom: 16,
    boxShadow: '0 0 24px rgba(240,192,64,0.06)',
  },
  th: {
    padding: '11px 16px',
    textAlign: 'left',
    fontSize: 11,
    color: 'var(--text3)',
    fontFamily: 'var(--mono)',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    borderBottom: '1px solid var(--border)',
    background: 'var(--bg4)',
    fontWeight: 500,
  },
  td: {
    padding: '11px 16px',
    fontSize: 14,
    borderBottom: '1px solid var(--border)',
    color: 'var(--text2)',
    fontFamily: 'var(--mono)',
  },
  tdPrimary: {
    padding: '11px 16px',
    fontSize: 14,
    borderBottom: '1px solid var(--border)',
    color: 'var(--text)',
    fontWeight: 500,
    fontFamily: 'var(--sans)',
  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function BoltIcon({ size = 16, color = 'var(--bolt)' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color} style={{ flexShrink: 0 }}>
      <path d="M13 2L4.5 13.5H11L10 22L20.5 10H14L13 2Z"/>
    </svg>
  )
}

function Spinner() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '28px 0', color: 'var(--text2)', fontFamily: 'var(--mono)', fontSize: 13 }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--bolt)" strokeWidth="2" style={{ animation: 'spin 1s linear infinite' }}>
        <circle cx="12" cy="12" r="10" strokeOpacity="0.2"/>
        <path d="M12 2a10 10 0 0 1 10 10"/>
      </svg>
      Analyse en cours - calcul des silences décisionnels…
    </div>
  )
}

function Badge({ value }) {
  const yes = value === true || value === 'true' || value === 'True'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 12px', borderRadius: 20, fontSize: 12,
      fontFamily: 'var(--mono)', fontWeight: 500,
      background: yes ? 'var(--green-bg)' : 'var(--red-bg)',
      color: yes ? 'var(--green)' : 'var(--red)',
      border: `1px solid ${yes ? 'rgba(61,220,132,0.2)' : 'rgba(255,85,85,0.2)'}`,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'currentColor' }}/>
      {yes ? 'Fin alerte' : 'Active'}
    </span>
  )
}

function ConfBadge({ val }) {
  const v = parseFloat(val)
  const color  = v >= THETA ? 'var(--green)' : v >= 0.5 ? 'var(--bolt)' : 'var(--red)'
  const bg     = v >= THETA ? 'var(--green-bg)' : v >= 0.5 ? 'var(--bolt-bg)' : 'var(--red-bg)'
  const border = v >= THETA ? 'rgba(61,220,132,0.2)' : v >= 0.5 ? 'rgba(240,192,64,0.2)' : 'rgba(255,85,85,0.2)'
  return (
    <span style={{
      display: 'inline-block', padding: '4px 12px', borderRadius: 20,
      fontSize: 12, fontFamily: 'var(--mono)', fontWeight: 500,
      background: bg, color, border: `1px solid ${border}`,
    }}>
      {v.toFixed(3)}
    </span>
  )
}

function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: 'var(--bg3)', border: '1px solid var(--border2)',
      borderRadius: 'var(--r)', padding: '20px 22px',
      borderTop: accent ? `2px solid ${accent}` : undefined,
    }}>
      <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>{label}</div>
      <div style={{ fontSize: 30, fontWeight: 300, color: 'var(--text)', fontFamily: 'var(--mono)', letterSpacing: '-0.02em' }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: 'var(--text2)', marginTop: 6 }}>{sub}</div>}
    </div>
  )
}

// Schéma CSV attendu
const CSV_SCHEMA = [
  { col: 'lightning_id',                  type: 'int',     desc: 'Identifiant unique de l\'éclair' },
  { col: 'lightning_airport_id',          type: 'int',     desc: 'Identifiant éclair côté aéroport' },
  { col: 'date',                          type: 'datetime',desc: 'Horodatage UTC (ex: 2023-01-08 22:23:16+00:00)' },
  { col: 'lon',                           type: 'float',   desc: 'Longitude (degrés décimaux)' },
  { col: 'lat',                           type: 'float',   desc: 'Latitude (degrés décimaux)' },
  { col: 'amplitude',                     type: 'float',   desc: 'Amplitude du signal (kA, peut être négative)' },
  { col: 'maxis',                         type: 'float',   desc: 'Indice de forme du signal' },
  { col: 'icloud',                        type: 'bool',    desc: 'True = intra-nuage (IC), False = nuage-sol (CG)' },
  { col: 'dist',                          type: 'float',   desc: 'Distance à l\'aéroport (km)' },
  { col: 'azimuth',                       type: 'float',   desc: 'Azimut depuis l\'aéroport (degrés)' },
  { col: 'airport',                       type: 'str',     desc: 'Nom de l\'aéroport (ex: Ajaccio)' },
  { col: 'airport_alert_id',              type: 'float',   desc: 'ID de l\'alerte associée (NaN si hors alerte)' },
  { col: 'is_last_lightning_cloud_ground',type: 'str',     desc: 'Variable cible - peut être vide pour la prédiction' },
]

function SchemaTable() {
  const typeColor = (t) => {
    if (t === 'int') return '#4a8fff'
    if (t === 'float') return '#a78bfa'
    if (t === 'bool') return '#3ddc84'
    if (t === 'datetime') return '#f0c040'
    return 'var(--text3)'
  }
  return (
    <div style={{ overflowX: 'auto', borderRadius: 'var(--r)', border: '1px solid var(--border)', marginTop: 16 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={c.th}>Colonne</th>
            <th style={c.th}>Type</th>
            <th style={c.th}>Description</th>
          </tr>
        </thead>
        <tbody>
          {CSV_SCHEMA.map((row, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? 'var(--bg2)' : 'var(--bg3)' }}>
              <td style={{ ...c.td, color: 'var(--bolt)', fontFamily: 'var(--mono)', fontSize: 12 }}>{row.col}</td>
              <td style={{ ...c.td, fontSize: 11 }}>
                <span style={{ color: typeColor(row.type), fontFamily: 'var(--mono)', fontSize: 11 }}>{row.type}</span>
              </td>
              <td style={{ ...c.td, color: 'var(--text2)', fontSize: 13 }}>{row.desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function UploadZone({ onFile, file }) {
  const [drag, setDrag] = useState(false)
  const ref = useRef()
  const handle = f => { if (f && (f.name.endsWith('.csv') || f.name.endsWith('.parquet'))) onFile(f) }

  return (
    <>
      <div
        onClick={() => ref.current.click()}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]) }}
        style={{
          border: `1.5px dashed ${drag ? 'var(--bolt)' : 'var(--border2)'}`,
          borderRadius: 'var(--r-lg)', padding: '36px 24px', textAlign: 'center',
          cursor: 'pointer', transition: 'all 0.2s',
          background: drag ? 'var(--bolt-bg)' : 'var(--bg4)',
          marginBottom: 14,
        }}
      >
        <div style={{ marginBottom: 12 }}>
          <BoltIcon size={30} color={drag ? 'var(--bolt)' : 'var(--text3)'} />
        </div>
        {file
          ? <div style={{ fontSize: 14, color: 'var(--bolt)', fontFamily: 'var(--mono)' }}>{file.name}</div>
          : <div style={{ fontSize: 14, color: 'var(--text2)' }}>Glissez votre fichier ici ou <span style={{ color: 'var(--bolt)' }}>parcourir</span></div>
        }
        <div style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--mono)', marginTop: 6 }}>
          CSV · PARQUET · 13 colonnes requises
        </div>
      </div>
      <input ref={ref} type="file" accept=".csv,.parquet" onChange={e => handle(e.target.files[0])} />
    </>
  )
}

function SummaryTable({ summary }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['Aéroport', 'Alerte ID', 'Décision θ=0.906', 'Confiance max', 'Dernière conf.'].map(h => (
              <th key={h} style={c.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {summary.map((row, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? 'var(--bg2)' : 'var(--bg3)' }}>
              <td style={c.tdPrimary}>{row.airport}</td>
              <td style={c.td}>{row.airport_alert_id}</td>
              <td style={c.td}><Badge value={row.end_alert} /></td>
              <td style={c.td}><ConfBadge val={row.conf_max} /></td>
              <td style={c.td}><ConfBadge val={row.conf_last} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page: Batch CSV (unique point d'entrée)
// ---------------------------------------------------------------------------

function BatchTab() {
  const [file, setFile]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [downloading, setDownloading] = useState(false)
  const [showSchema, setShowSchema] = useState(false)

  const handleAnalyse = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await fetch(`${API}/predict/summary`, { method: 'POST', body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Erreur serveur') }
      setResult(await res.json())
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  const handleDownload = async () => {
    if (!file) return
    setDownloading(true)
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await fetch(`${API}/predict/csv`, { method: 'POST', body: fd })
      const blob = await res.blob()
      const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'predictions.csv' })
      a.click(); URL.revokeObjectURL(a.href)
    } finally { setDownloading(false) }
  }

  const nFin = result?.summary?.filter(r => r.end_alert === true || r.end_alert === 'true' || r.end_alert === 'True').length ?? 0

  return (
    <div>
      <div style={c.card}>
        <div style={{ padding: '22px 26px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontSize: 17, fontWeight: 500, color: 'var(--text)', marginBottom: 6 }}>Import des données foudre</div>
          <div style={{ fontSize: 15, color: 'var(--text2)' }}>
            Importez un CSV ou Parquet de données brutes éclairs. Le pipeline construit automatiquement
            les silences décisionnels et calcule P(fin d'alerte) pour chaque alerte détectée. Si cette
            probabilité est supérieure à un seuil θ, l'alerte est levée, sinon elle reste active.
          </div>
        </div>
        <div style={{ padding: '22px 26px' }}>
          <UploadZone onFile={setFile} file={file} />

          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 16 }}>
            <button
              onClick={handleAnalyse} disabled={!file || loading}
              style={{
                padding: '11px 24px', borderRadius: 'var(--r)', fontSize: 14, fontWeight: 500,
                cursor: file && !loading ? 'pointer' : 'not-allowed', border: 'none',
                fontFamily: 'var(--sans)', background: file && !loading ? 'var(--bolt)' : 'var(--bg4)',
                color: file && !loading ? '#000' : 'var(--text3)', transition: 'all 0.2s',
              }}
            >
              {loading ? 'Calcul…' : 'Lancer l\'analyse'}
            </button>
            {result && (
              <button
                onClick={handleDownload} disabled={downloading}
                style={{
                  padding: '11px 24px', borderRadius: 'var(--r)', fontSize: 14, fontWeight: 500,
                  cursor: 'pointer', fontFamily: 'var(--sans)',
                  background: 'transparent', color: 'var(--text2)',
                  border: '1px solid var(--border2)', transition: 'all 0.15s',
                }}
              >
                {downloading ? 'Export…' : 'Télécharger predictions.csv'}
              </button>
            )}
            <button
              onClick={() => setShowSchema(s => !s)}
              style={{
                padding: '11px 20px', borderRadius: 'var(--r)', fontSize: 13, fontWeight: 400,
                cursor: 'pointer', fontFamily: 'var(--mono)',
                background: 'transparent', color: 'var(--text3)',
                border: '1px solid var(--border)', transition: 'all 0.15s',
              }}
            >
              {showSchema ? 'Masquer' : 'Format CSV attendu'}
            </button>
          </div>

          {showSchema && <SchemaTable />}
        </div>
      </div>

      {loading && <Spinner />}
      {error && (
        <div style={{ padding: '14px 18px', borderRadius: 'var(--r)', background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid rgba(255,85,85,0.2)', fontFamily: 'var(--mono)', fontSize: 13, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {result && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
            <MetricCard label="Éclairs chargés"      value={result.meta.n_lightnings.toLocaleString('fr-FR')} accent="var(--blue)" />
            <MetricCard label="Aéroports"             value={result.meta.n_airports} accent="var(--blue)" />
            <MetricCard label="Alertes analysées"     value={result.n_alerts} accent="var(--bolt)" />
            <MetricCard label="Fin alerte détectée"   value={nFin} sub={`sur ${result.n_alerts} alertes`} accent="var(--green)" />
          </div>

          <div style={c.cardGlow}>
            <div style={{ padding: '18px 26px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: 15, fontWeight: 500, color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <BoltIcon size={15} /> Résultats par alerte
              </div>
              <div style={{ fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--text3)' }}>
                seuil θ = {THETA}
              </div>
            </div>
            <SummaryTable summary={result.summary} />
          </div>
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page: Modèle
// ---------------------------------------------------------------------------

const FEATURE_GROUPS = [
  {
    label: 'Durée & temporel',
    color: 'var(--bolt)',
    features: [
      { name: 'elapsed_min',                desc: 'Minutes écoulées depuis le début de l\'alerte' },
      { name: 'obs_elapsed_min',            desc: 'Minutes écoulées depuis le début de la fenêtre d\'observation' },
      { name: 'minutes_since_reference_cg', desc: 'Minutes depuis le dernier CG inner de référence' },
    ],
  },
  {
    label: 'Comptages globaux',
    color: 'var(--blue)',
    features: [
      { name: 'n_total',      desc: 'Nombre total d\'éclairs observés depuis obs_start' },
      { name: 'n_cg',         desc: 'Nombre d\'éclairs nuage-sol (CG)' },
      { name: 'n_ic',         desc: 'Nombre d\'éclairs intra-nuage (IC)' },
      { name: 'n_inner',      desc: 'Nombre d\'éclairs dans la zone inner (< 20 km)' },
      { name: 'n_outer',      desc: 'Nombre d\'éclairs dans la zone outer (≥ 20 km)' },
      { name: 'n_cg_inner',   desc: 'Éclairs CG dans la zone inner' },
      { name: 'n_cg_outer',   desc: 'Éclairs CG dans la zone outer' },
      { name: 'n_ic_inner',   desc: 'Éclairs IC dans la zone inner' },
      { name: 'n_ic_outer',   desc: 'Éclairs IC dans la zone outer' },
      { name: 'n_lre',        desc: 'Nombre d\'éclairs très proches (< 3 km, LRE)' },
    ],
  },
  {
    label: 'Amplitude & distance',
    color: '#a78bfa',
    features: [
      { name: 'amp_abs_mean',       desc: 'Amplitude absolue moyenne de tous les éclairs' },
      { name: 'amp_abs_max',        desc: 'Amplitude absolue maximale' },
      { name: 'dist_mean',          desc: 'Distance moyenne à l\'aéroport (km)' },
      { name: 'dist_min',           desc: 'Distance minimale à l\'aéroport (km)' },
      { name: 'last_event_amplitude', desc: 'Amplitude du dernier éclair observé' },
      { name: 'last_event_dist',    desc: 'Distance du dernier éclair observé (km)' },
      { name: 'last_event_type',    desc: 'Type du dernier éclair (CG / IC)' },
      { name: 'last_event_zone',    desc: 'Zone du dernier éclair (inner / outer)' },
    ],
  },
  {
    label: 'Temps depuis dernier éclair',
    color: 'var(--green)',
    features: [
      { name: 'time_since_last_event_min',     desc: 'Silence depuis le tout dernier éclair (min)' },
      { name: 'time_since_last_cg_min',        desc: 'Silence depuis le dernier CG (min)' },
      { name: 'time_since_last_ic_min',        desc: 'Silence depuis le dernier IC (min)' },
      { name: 'time_since_last_inner_min',     desc: 'Silence depuis le dernier éclair inner (min)' },
      { name: 'time_since_last_cg_inner_min',  desc: 'Silence depuis le dernier CG inner (min) - feature clé' },
    ],
  },
  {
    label: 'Inter-arrivées',
    color: 'var(--bolt)',
    features: [
      { name: 'mean_interarrival_min',                        desc: 'Temps moyen entre deux éclairs consécutifs (min)' },
      { name: 'median_interarrival_min',                      desc: 'Temps médian entre deux éclairs (min)' },
      { name: 'max_interarrival_min',                         desc: 'Plus grand écart entre deux éclairs (min)' },
      { name: 'current_silence_over_mean_interarrival',       desc: 'Silence actuel / inter-arrivée moyenne' },
      { name: 'current_silence_over_median_interarrival',     desc: 'Silence actuel / inter-arrivée médiane' },
      { name: 'current_silence_over_max_interarrival',        desc: 'Silence actuel / inter-arrivée maximale' },
    ],
  },
  {
    label: 'Fenêtres glissantes (5 · 10 · 20 min)',
    color: 'var(--blue)',
    features: [
      { name: 'n_total_last_{w}m',      desc: 'Nb total d\'éclairs sur la fenêtre' },
      { name: 'n_cg_inner_last_{w}m',   desc: 'Nb CG inner sur la fenêtre' },
      { name: 'n_cg_outer_last_{w}m',   desc: 'Nb CG outer sur la fenêtre' },
      { name: 'n_ic_inner_last_{w}m',   desc: 'Nb IC inner sur la fenêtre' },
      { name: 'n_ic_outer_last_{w}m',   desc: 'Nb IC outer sur la fenêtre' },
      { name: 'n_lre_last_{w}m',        desc: 'Nb éclairs LRE (< 3 km) sur la fenêtre' },
      { name: 'dist_mean_last_{w}m',    desc: 'Distance moyenne sur la fenêtre (km)' },
      { name: 'dist_min_last_{w}m',     desc: 'Distance minimale sur la fenêtre (km)' },
      { name: 'amp_abs_mean_last_{w}m', desc: 'Amplitude absolue moyenne sur la fenêtre' },
    ],
  },
  {
    label: 'LRE & approche progressive',
    color: 'var(--red)',
    features: [
      { name: 'has_lre_before',         desc: 'Au moins un éclair à < 3 km a été observé (0/1)' },
      { name: 'time_since_last_lre_min',desc: 'Silence depuis le dernier éclair LRE (< 3 km)' },
      { name: 'delta_dist_min_20_5',    desc: 'dist_min_last_20m − dist_min_last_5m (approche)' },
      { name: 'delta_dist_mean_20_5',   desc: 'dist_mean_last_20m − dist_mean_last_5m' },
      { name: 'n_lt_3km_last_10m',      desc: 'Nb éclairs à < 3 km sur les 10 dernières minutes' },
      { name: 'n_lt_3km_last_20m',      desc: 'Nb éclairs à < 3 km sur les 20 dernières minutes' },
    ],
  },
]

const PARAMS = [
  ['Algorithme',        'XGBoost (binary:logistic)'],
  ['n_estimators',      '100'],
  ['max_depth',         '7'],
  ['learning_rate',     '0.03'],
  ['subsample',         '0.8'],
  ['colsample_bytree',  '0.8'],
  ['min_child_weight',  '1'],
  ['reg_lambda',        '1.0'],
  ['Seuil θ',           '0.906'],
  ['Entraînement',      '100% des données'],
  ['Préprocessing',     'SimpleImputer · OneHotEncoder'],
]

function ModelPage() {
  const totalFeatures = FEATURE_GROUPS.reduce((a, g) => a + g.features.length, 0)
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={c.card}>
          <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--border)', fontSize: 17, fontWeight: 500, color: 'var(--text)' }}>
            Objectif
          </div>
          <div style={{ padding: '18px 22px', fontSize: 15, color: 'var(--text2)', lineHeight: 1.9 }}>
            <p>Prédire si, à un instant <code style={{ fontFamily: 'var(--mono)', color: 'var(--bolt)', fontSize: 15 }}>t</code> de silence,
            il n'y aura plus aucun éclair CG (nuage→sol) dans la zone inner (&lt; 20 km) après <code style={{ fontFamily: 'var(--mono)', color: 'var(--bolt)', fontSize: 15 }}>t</code>.</p>
            <br/>
            <p><code style={{ fontFamily: 'var(--mono)', color: 'var(--bolt)', fontSize: 15 }}>y = 1</code> → alerte terminée,
            &nbsp;<code style={{ fontFamily: 'var(--mono)', color: 'var(--text3)', fontSize: 15 }}>y = 0</code> → alerte encore active.</p>
            <br/>
            <p>Instants de décision générés toutes les <strong style={{ color: 'var(--text)' }}>3 min</strong> de silence,
            jusqu'à <strong style={{ color: 'var(--text)' }}>30 min</strong> après le dernier éclair CG (Baseline).</p>
          </div>
        </div>

        <div style={c.card}>
          <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--border)', fontSize: 17, fontWeight: 500, color: 'var(--text)' }}>
            Hyperparamètres
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <tbody>
              {PARAMS.map(([k, v], i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? 'var(--bg2)' : 'var(--bg3)' }}>
                  <td style={{ ...c.td, color: 'var(--text3)', fontSize: 12 }}>{k}</td>
                  <td style={{ ...c.td, color: 'var(--bolt)', fontSize: 13 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={c.card}>
        <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--border)', fontSize: 15, fontWeight: 500, color: 'var(--text)' }}>
          Features utilisées - 74 features au total (dont 27 via fenêtres glissantes ×3)
        </div>
        <div style={{ padding: '18px 22px', display: 'flex', flexDirection: 'column', gap: 24 }}>
          {FEATURE_GROUPS.map((group, gi) => (
            <div key={gi}>
              <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color: group.color, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10, fontWeight: 500 }}>
                {group.label}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0, border: '1px solid var(--border)', borderRadius: 'var(--r)', overflow: 'hidden' }}>
                {group.features.map((f, fi) => (
                  <div key={fi} style={{
                    padding: '9px 14px',
                    borderBottom: fi < group.features.length - 2 ? '1px solid var(--border)' : 'none',
                    borderRight: fi % 2 === 0 ? '1px solid var(--border)' : 'none',
                    background: fi % 4 < 2 ? 'var(--bg2)' : 'var(--bg3)',
                    display: 'flex', flexDirection: 'column', gap: 2,
                  }}>
                    <code style={{ fontSize: 11, fontFamily: 'var(--mono)', color: group.color }}>{f.name}</code>
                    <span style={{ fontSize: 12, color: 'var(--text2)' }}>{f.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page: À propos
// ---------------------------------------------------------------------------

function AboutPage() {
  const team = [
    { name: 'Mehdi NEJI' },
    { name: 'Ali BEN HADJ YAHIA' },
    { name: 'Lina GAROUACHI' },
  ]

  return (
    <div>
      <div style={c.card}>
        <div style={{ padding: '18px 24px', borderBottom: '1px solid var(--border)', fontSize: 15, fontWeight: 500, color: 'var(--text)' }}>
          Équipe · Data Battle 2026
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1, background: 'var(--border)' }}>
          {team.map((m, i) => (
            <div key={i} style={{ background: 'var(--bg2)', padding: '28px 20px', textAlign: 'center' }}>
              <div style={{
                width: 52, height: 52, borderRadius: '50%', margin: '0 auto 14px',
                background: 'var(--bg4)', border: '1px solid var(--border2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 18, color: 'var(--bolt)', fontWeight: 500,
              }}>
                {m.name.charAt(0)}
              </div>
              <div style={{ fontSize: 15, fontWeight: 500, color: 'var(--text)' }}>{m.name}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={c.card}>
        <div style={{ padding: '18px 24px', borderBottom: '1px solid var(--border)', fontSize: 15, fontWeight: 500, color: 'var(--text)' }}>
          Stack technique
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1, background: 'var(--border)' }}>
          {[
            ['Modèle',   'XGBoost · sklearn pipeline', 'var(--bolt)'],
            ['Backend',  'FastAPI · uvicorn',           'var(--blue)'],
            ['Frontend', 'React · Vite',                'var(--green)'],
            ['Features', '~50 features · silences décisionnels', 'var(--bolt)'],
            ['Données',  'Météorage · 5 aéroports',     'var(--blue)'],
            ['Seuil',    'θ = 0.906 · optimisé sur train', 'var(--green)'],
          ].map(([title, desc, color], i) => (
            <div key={i} style={{ background: 'var(--bg2)', padding: '18px 22px' }}>
              <div style={{ fontSize: 11, fontFamily: 'var(--mono)', color, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>{title}</div>
              <div style={{ fontSize: 13, color: 'var(--text2)' }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page: RSE (à compléter)
// ---------------------------------------------------------------------------

function RSEPage() {
  return (
    <div>
      <div style={{ ...c.card, border: '1px dashed var(--border2)' }}>
        <div style={{ padding: '40px 28px', textAlign: 'center' }}>
          <div style={{ fontSize: 13, fontFamily: 'var(--mono)', color: 'var(--text3)', marginBottom: 12 }}>
            Section RSE - à compléter
          </div>
          <div style={{ fontSize: 14, color: 'var(--text3)', lineHeight: 1.8, maxWidth: 480, margin: '0 auto' }}>
            Cette section accueillera les outils de calcul d'impact environnemental et social du projet :
            consommation énergétique du modèle, empreinte carbone des prédictions, effets rebonds estimés.
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

const NAV = [
  { id: 'batch',  label: 'Prédiction',  color: 'var(--bolt)' },
  { id: 'model',  label: 'Modèle',      color: 'var(--blue)' },
  { id: 'about',  label: 'À propos',    color: 'var(--text3)' },
  { id: 'rse',    label: 'RSE',         color: 'var(--green)' },
]

const PAGE_TITLES = {
  batch:  { title: 'Prédiction de fin d\'alerte', sub: 'Import CSV ou Parquet · calcul des silences décisionnels · XGBoost' },
  model:  { title: 'Modèle XGBoost',              sub: 'Features · hyperparamètres · pipeline' },
  about:  { title: 'À propos',                    sub: 'Data Battle 2026 · IA Pau × Météorage' },
  rse:    { title: 'Responsabilité sociale',       sub: 'Impact environnemental et social du projet' },
}

export default function App() {
  const [tab, setTab] = useState('batch')
  const { title, sub } = PAGE_TITLES[tab]

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)' }}>

      <aside style={{
        width: 216, background: 'var(--bg2)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        padding: '22px 16px', flexShrink: 0,
      }}>
        <div style={{ marginBottom: 30, paddingBottom: 22, borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 5 }}>
            <div style={{ animation: 'pulse-bolt 2s ease-in-out infinite' }}>
              <BoltIcon size={19} color="var(--bolt)" />
            </div>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 14, fontWeight: 500, color: 'var(--text)', letterSpacing: '0.04em' }}>
              DATAVERSE
            </span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text3)', paddingLeft: 28 }}>
            Data Battle 2026
          </div>
        </div>

        <nav style={{ flex: 1 }}>
          {NAV.map(n => (
            <div
              key={n.id}
              onClick={() => setTab(n.id)}
              style={{
                padding: '10px 14px', borderRadius: 'var(--r)', fontSize: 14,
                cursor: 'pointer', marginBottom: 3,
                display: 'flex', alignItems: 'center', gap: 10,
                background: tab === n.id ? 'var(--bg4)' : 'transparent',
                color: tab === n.id ? 'var(--text)' : 'var(--text2)',
                border: `1px solid ${tab === n.id ? 'var(--border2)' : 'transparent'}`,
                transition: 'all 0.15s', fontWeight: tab === n.id ? 500 : 400,
              }}
            >
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: n.color, flexShrink: 0 }} />
              {n.label}
            </div>
          ))}
        </nav>

        <div style={{ paddingTop: 18, borderTop: '1px solid var(--border)', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text3)', lineHeight: 2 }}>
          <div>XGBoost · sklearn</div>
          <div>θ = 0.906</div>
          <div style={{ marginTop: 8, fontSize: 11 }}>IA Pau × Météorage</div>
        </div>
      </aside>

      <main style={{ flex: 1, overflow: 'auto', padding: '30px 36px' }}>
        <div style={{ marginBottom: 26 }}>
          <h1 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text)', marginBottom: 6, letterSpacing: '-0.01em' }}>
            {title}
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text2)', fontFamily: 'var(--mono)' }}>{sub}</p>
        </div>

        {tab === 'batch' && <BatchTab />}
        {tab === 'model' && <ModelPage />}
        {tab === 'about' && <AboutPage />}
        {tab === 'rse'   && <RSEPage />}
      </main>
    </div>
  )
}