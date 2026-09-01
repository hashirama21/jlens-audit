import os, urllib.request, json, time
from src import conditions


def usage():
    k = os.environ['OPENROUTER_API_KEY']
    r = urllib.request.Request('https://openrouter.ai/api/v1/credits',
                               headers={'Authorization': f'Bearer {k}'})
    return json.load(urllib.request.urlopen(r, timeout=30))['data']['total_usage']


def log(m):
    print('[' + time.strftime('%H:%M:%S') + '] ' + m, flush=True)


log('START usage=$%.4f' % usage())
log('=== BLOC E: jlens (v1+v2, 2 juges) ===')
conditions.run('pairs.jsonl', instruments=['jlens'])
uE = usage()
log('FIN E  usage=$%.4f' % uE)
log('=== BLOC F: rlens+logit (v1, 2 juges) ===')
conditions.run('pairs.jsonl', instruments=['rlens', 'logit'], prompt_versions=['v1'])
uF = usage()
log('FIN F  usage=$%.4f  restant=$%.4f' % (uF, 19 - uF))
log('DONE')
