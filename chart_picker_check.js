// Self-check for the dashboard's overview chart picker. Run: node chart_picker_check.js
const fs = require('fs'), A = require('assert');
const html = fs.readFileSync(__dirname + '/dashboard.html', 'utf8');
eval(html.slice(html.indexOf('  var SKIP_NUMERIC'), html.indexOf('  // ---- data fetchers')));

const cats = ['Clothing', 'Footwear', 'Bags'];
const shop = [...Array(60)].map((_, i) => ({ id: i, product_name: 'P' + i, category: cats[i % 3], selling_price: 100 + i }));
A.deepStrictEqual(planCharts(Object.keys(shop[0]), shop), { shape: 'groups', group: 'category', measure: 'selling_price' });

const days = [...Array(30)].map((_, i) => `${String(1 + i).padStart(2, '0')}-01-2024`);   // day-first, like NSE exports
const stock = days.map((d, i) => ({ id: i, price_date: d, open_price: 1500 + i, close_price: 1505 + i, traded_quantity: 1e6 + i }));
A.deepStrictEqual(planCharts(Object.keys(stock[0]), stock),
  { shape: 'time', date: 'price_date', measure: 'close_price', group: undefined, measure2: 'traded_quantity' });

const wide = ['Revenues', 'Cost', 'Margin %'].map((n, i) => ({ id: i, line_item: n, q1_24: '100', q2_24: 110, q3_24: 120, q4_24: 130, fy_24: 460 }));
A.deepStrictEqual(planCharts(Object.keys(wide[0]), wide), { shape: 'wide', label: 'line_item', periods: ['q1_24', 'q2_24', 'q3_24', 'q4_24', 'fy_24'] });

const facts = [];   // start is mostly blank; the chart must use the complete 'end' column
['Revenue', 'Profit', 'Equity'].forEach(c => [2021, 2022, 2023, 2024].forEach(y =>
  facts.push({ id: facts.length, concept: c, unit: 'USD', start: c === 'Equity' ? null : `${y - 1}-04-01`, end: `${y}-03-31`, value: y * 10 })));
A.deepStrictEqual(planCharts(Object.keys(facts[0]), facts), { shape: 'long', date: 'end', key: 'concept', measure: 'value' });

const pages = [...Array(5)].map((_, i) => ({ id: i, page_number: i + 1, text: 'word '.repeat(80 + i), has_text: true }));
A.strictEqual(planCharts(Object.keys(pages[0]), pages).shape, 'pages');
A.strictEqual(planCharts(['id', 'body'], [...Array(20)].map((_, i) => ({ id: i, body: 'note ' + i }))).shape, 'none');
A.strictEqual(pickHeadline(['Gross Profit', 'Revenues']), 'Revenues');
A.strictEqual(dateTime('13/02/2025') > dateTime('12/02/2025'), true);
console.log('chart picker checks OK');
