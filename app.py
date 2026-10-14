import re, sqlite3, secrets, hashlib, hmac, datetime, html
from pathlib import Path
import streamlit as st
import joblib

BASE=Path(__file__).parent
DB=BASE/'student_cases.sqlite3'
TASHKEEL=re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')
def normalize_arabic(text):
    text=TASHKEEL.sub('',str(text)).replace('ـ','')
    text=re.sub('[إأآٱ]','ا',text).replace('ى','ي')
    return re.sub(r'\s+',' ',text).strip()

PREFIX=[r'^أنا لاحظت إن\s*',r'^لما رجعت أتأكد لقيت إن\s*',r'^كنت بتابع الموضوع ولقيت إن\s*',r'^اللي حصل معايا إن\s*',r'^محتاج توضيح بخصوص الآتي:\s*',r'^عندي مشكلة وعايز حل:\s*',r'^عايز أعرف أتصرف إزاي[،,:]\s*',r'^عايز أعرف سبب إن\s*',r'^ممكن توضحوا موضوع إن\s*',r'^أنا مش فاهم ليه\s*',r'^ممكن أفهم ليه\s*',r'^هو المفروض أعمل إيه لو\s*',r'^هو ليه\s*',r'^بستفسر عن حاجة[،,:]\s*',r'^كنت عايز أسأل:\s*',r'^أنا محتاج مساعدة[،,:]\s*',r'^ممكن حد يفهمني[،,:]\s*',r'^يا جماعة\s*',r'^لو سمحت[،,:]?\s*',r'^من فضلكم[،,:]?\s*',r'^بعد إذنكم[،,:]?\s*',r'^برجاء الإفادة[،,:]?\s*',r'^برجاء المساعدة[،,:]?\s*',r'^أرجو المساعدة[،,:]?\s*',r'^أود الاستفسار[،,:]?\s*',r'^السلام عليكم[،,:]?\s*',r'^معلش[،,:]?\s*']
CLOSINGS=['ممكن مساعدة','أرجو مراجعة الموضوع','يا ريت تراجعوا ده','محتاج أعرف الخطوة الجاية','محتاج الموضوع يتراجع','محتاج رد لو سمحتم','أعمل إيه دلوقتي','أتصرف إزاي','ممكن توضيح','هل فيه حل متاح','ممكن أعرف الحل','مش عارف أتصرف إزاي','ومش عارف أبدأ منين','وأتمنى مراجعة الأمر','وأرجو توضيح الإجراء المناسب','ممكن تساعدوني أفهم السبب','يا ريت حد يوضح لي الخطوة الجاية','أتواصل مع مين بخصوص ده','هل في طريقة أتابع بيها','عايز أتأكد إن الموضوع اتسجل']
FILLERS=['ما عنديش معلومات زيادة غير اللي ظاهر قدامي','أنا مش متأكد إذا الخطأ من عندي ولا من النظام','مش عايز أعمل خطوة غلط وأضيع وقت','سألت قبل كده ولسه ما وصلت لرد واضح','جربت أستفسر بس الردود كانت مختلفة','كلمت حد وقال لي أتابع بعدين من غير تفاصيل','حاولت أراجع بنفسي لكن ما فهمتش اللي حصل','دخلت على حسابي تاني ونفس الوضع موجود','يهمني أعرف هل لازم أقدم طلب جديد','الموضوع مقلقني لأن عندي إجراء تاني مرتبط بيه','وده معطلني في اللي ورايا']
def derive(s):
    t=s.strip()
    for p in PREFIX:
        n=re.sub(p,'',t,count=1)
        if n!=t:t=n;break
    for f in FILLERS:t=re.sub(r'(?:[.؟!]\s*|،\s*)'+re.escape(f)+r'(?=[.؟!،]|$)','',t)
    for c in CLOSINGS:t=re.sub(r'(?:[،.؟!]\s*|\s+)'+re.escape(c)+r'[؟.!\s]*$','',t)
    t=re.sub(r'\s+',' ',t).strip(' ،.؟!:')
    if re.search(r'(?:مش عارف|مش قادر|ومحتاج|ومش|وعايز|لكن|بس)$',t):return s.strip()
    if not t or len(t)<12 or len(t)>len(s):return s.strip()
    return t

@st.cache_resource(show_spinner=False)
def models():
    for name in ('classifier.joblib','compression.joblib'):
        if not (BASE/name).is_file():raise FileNotFoundError(name)
    # Classifier artifact references normalize_arabic in __main__ (training-time pickle).
    # Expose the exact original preprocessing callable before trusted artifact loading.
    import __main__
    __main__.normalize_arabic=normalize_arabic
    clf=joblib.load(BASE/'classifier.joblib')
    cmp=joblib.load(BASE/'compression.joblib')
    if not hasattr(clf,'predict') or not isinstance(cmp,dict) or not {'vectorizer','model'}<=cmp.keys():
        raise ValueError('Unexpected model artifact contract')
    return clf,cmp
UNITS={'CARDS':('Cards Office','وحدة الكارنيهات'),'EXAMS':('Examinations Office','وحدة الامتحانات والكونترول'),'RECORDS':('Records & Registration','وحدة القيد والملفات'),'RESULTS':('Results Office','وحدة النتائج')}
def infer(body):
    clf,cmp=models()
    label=str(clf.predict([body])[0])
    if label not in UNITS:raise ValueError('Unmapped classifier label: '+label)
    decision=cmp['model'].predict(cmp['vectorizer'].transform([body]))[0]
    if int(decision) not in (0,1):raise ValueError('Unexpected compression decision')
    summary=derive(body) if int(decision)==1 else body
    return label,summary,bool(decision)
def connect():
    c=sqlite3.connect(DB,timeout=15);c.row_factory=sqlite3.Row;return c
def init():
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS cases (id INTEGER PRIMARY KEY, reference TEXT UNIQUE, code_hash TEXT, name TEXT, email TEXT, body TEXT, summary TEXT, unit TEXT, compression INTEGER, created_at TEXT, rating INTEGER, comment TEXT, request_id TEXT UNIQUE)')
        cols={r[1] for r in c.execute('PRAGMA table_info(cases)')}
        if 'request_id' not in cols:c.execute('ALTER TABLE cases ADD COLUMN request_id TEXT')
        c.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, case_id INTEGER, action TEXT, note TEXT, created_at TEXT)')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def save(name,email,body,label,summary,compressed,request_id):
    ref='SD-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')+'-'+secrets.token_hex(5).upper()
    code=secrets.token_urlsafe(18)
    with connect() as c:
        existing=c.execute('SELECT reference FROM cases WHERE request_id=?',(request_id,)).fetchone()
        if existing:return existing['reference'],None
        cur=c.execute('INSERT INTO cases(reference,code_hash,name,email,body,summary,unit,compression,created_at,request_id) VALUES (?,?,?,?,?,?,?,?,?,?)',(ref,hashlib.sha256(code.encode()).hexdigest(),name,email,body,summary,label,int(compressed),now(),request_id))
        c.execute('INSERT INTO events(case_id,action,note,created_at) VALUES (?,?,?,?)',(cur.lastrowid,'Submitted','Submission recorded; destination suggested, not assigned.',now()))
    return ref,code
def lookup(ref,code):
    if not ref.strip() or not code:return None,None
    with connect() as c:
        row=c.execute('SELECT * FROM cases WHERE reference=?',(ref.strip(),)).fetchone()
        if row and hmac.compare_digest(row['code_hash'],hashlib.sha256(code.encode()).hexdigest()):
            return dict(row),[dict(e) for e in c.execute('SELECT * FROM events WHERE case_id=? ORDER BY id',(row['id'],))]
    return None,None

I={
'en':{'overview':'Overview','submit':'Submit Complaint','track':'Track Complaint','feedback':'Service Feedback','welcome':'STUDENT SERVICES','headline':'A better way to be heard.','intro':'Share your concern in your own words, review the details and keep your reference for future tracking.','start':'Start a complaint','tracknow':'Track a complaint','rate':'Share feedback','journey':'How it works','step1':'Describe your concern','step2':'Review the proposed details','step3':'Confirm and receive your reference','step4':'Track and rate your experience','note':'This service records submissions. The proposed destination is not a confirmed administrative assignment.','formtitle':'Tell us what happened','formintro':'No need to choose a category or department. Please write in Arabic for the current trained models.','name':'Full name','email':'Email address','body':'Your complaint','placeholder':'Explain what happened, when it happened and how it affected you...','analyze':'Review complaint','review':'Review your complaint','original':'Original complaint','summary':'Suggested summary','destination':'Proposed responsible department','proposed':'Suggested only — not yet assigned','edit':'Edit complaint','confirm':'Confirm and submit','success':'Your complaint has been recorded','receiptinfo':'Save both values securely. You will need them to retrieve your submission.','reference':'Complaint reference','code':'Private tracking code','timestamp':'Submitted at (UTC)','trackingtitle':'Track your submission','trackingintro':'Enter your reference and private tracking code to view recorded activity.','find':'Find complaint','status':'Current status','received':'Received — awaiting institutional processing','timeline':'Recorded activity','invalid':'Reference or tracking code is incorrect.','feedbacktitle':'Rate your experience','feedbackintro':'Tell us how easy it was to submit and track your complaint. This does not rate problem resolution.','rating':'Submission experience (1–5)','comment':'Additional comments (optional)','sendrating':'Send feedback','thanks':'Thank you. Your feedback has been recorded.','already':'Feedback has already been submitted for this complaint.','required':'Please enter your name, a valid email and a complaint of at least 12 characters.','failure':'We could not process your complaint. Nothing was submitted. Please try again.','noactivity':'Only submission events are recorded in this version.','language':'Language','private':'Keep your tracking code private','new':'New complaint','loading':'Processing your complaint...','copy':'Copy using the code field controls','summaryhelp':'You may adjust the proposed summary without changing the original complaint.','saved':'Submission recorded','unavailable':'This installation does not provide staff processing or resolution updates.','ratefirst':'Use your complaint reference and private code to leave feedback.'},
'ar':{'overview':'الرئيسية','submit':'تقديم شكوى','track':'متابعة الشكوى','feedback':'تقييم التجربة','welcome':'خدمات الطلاب','headline':'صوتك مهم، وخطواتك واضحة.','intro':'احكي مشكلتك بطريقتك، راجع التفاصيل، واحتفظ برقم المتابعة للرجوع لشكواك.','start':'تقديم شكوى جديدة','tracknow':'متابعة شكوى','rate':'شاركنا تقييمك','journey':'رحلتك مع الشكوى','step1':'اكتب شكواك بحرية','step2':'راجع البيانات المقترحة','step3':'أكد الإرسال واستلم رقم المتابعة','step4':'تابع شكواك وقيّم التجربة','note':'الخدمة تسجل الشكوى، والجهة المعروضة مقترحة وليست تكليفًا إداريًا مؤكدًا.','formtitle':'احكي لنا اللي حصل','formintro':'مش محتاج تختار نوع الشكوى أو الإدارة. اكتب شكواك بالعربي.','name':'الاسم الكامل','email':'البريد الإلكتروني','body':'تفاصيل الشكوى','placeholder':'احكي اللي حصل وإمتى حصل وإزاي أثر عليك...','analyze':'مراجعة الشكوى','review':'راجع شكواك قبل الإرسال','original':'نص الشكوى الأصلي','summary':'الملخص المقترح','destination':'الإدارة المختصة المقترحة','proposed':'اقتراح فقط — لم يتم التكليف رسميًا','edit':'تعديل الشكوى','confirm':'تأكيد وإرسال الشكوى','success':'تم تسجيل شكواك بنجاح','receiptinfo':'احتفظ بالرقم والرمز السري في مكان آمن، لأنك هتحتاجهم للمتابعة.','reference':'رقم الشكوى','code':'رمز المتابعة السري','timestamp':'وقت التقديم (UTC)','trackingtitle':'متابعة الشكوى','trackingintro':'اكتب رقم الشكوى ورمز المتابعة السري لعرض الأحداث المسجلة.','find':'عرض الشكوى','status':'الحالة الحالية','received':'تم الاستلام — بانتظار المعالجة المؤسسية','timeline':'سجل الأحداث','invalid':'رقم الشكوى أو رمز المتابعة غير صحيح.','feedbacktitle':'قيّم تجربتك','feedbackintro':'قيّم سهولة تقديم الشكوى ومتابعتها، وليس حل المشكلة نفسها.','rating':'تقييم تجربة التقديم (١–٥)','comment':'ملاحظات إضافية (اختياري)','sendrating':'إرسال التقييم','thanks':'شكرًا لك، تم تسجيل تقييمك.','already':'تم إرسال تقييم لهذه الشكوى من قبل.','required':'من فضلك اكتب اسمك وبريد صحيح وشكوى لا تقل عن ١٢ حرفًا.','failure':'تعذر معالجة الشكوى ولم يتم إرسالها. حاول مرة أخرى.','noactivity':'هذه النسخة تسجل أحداث التقديم فقط.','language':'اللغة','private':'احتفظ برمز المتابعة بسرية','new':'شكوى جديدة','loading':'جاري معالجة الشكوى...','copy':'يمكنك نسخ البيانات من حقول الكود','summaryhelp':'تقدر تعدل الملخص المقترح بدون تغيير نص الشكوى الأصلي.','saved':'تم تسجيل الشكوى','unavailable':'هذه النسخة لا تشمل إجراءات الموظفين أو تحديثات الحل.','ratefirst':'استخدم رقم الشكوى والرمز السري لتقييم التجربة.'}}

def t(k):return I[st.session_state.lang][k]
def go(p):st.session_state.page=p;st.rerun()
def panel(title,body='',kind='normal'):
    st.markdown(f'<div class="sadat-panel {kind}"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',unsafe_allow_html=True)

st.set_page_config(page_title='SADAT | Student Services',page_icon=str(BASE/'logo.jpg'),layout='wide',initial_sidebar_state='expanded')
init()
for key,value in {'lang':'en','page':'Overview','draft':None,'receipt':None,'request_id':None,'tracking_result':None,'rating_done':False}.items():st.session_state.setdefault(key,value)
css=(BASE/'assets'/'theme.css').read_text(encoding='utf-8')
st.markdown('<style>'+css+'</style>',unsafe_allow_html=True)
with st.sidebar:
    st.image(str(BASE/'logo.jpg'),width=76)
    st.markdown('<div class="sadat-brand">SADAT</div><div class="sadat-brand-sub">SERVICE QUALITY CENTER</div>',unsafe_allow_html=True)
    st.divider()
    pages=['Overview','Submit Complaint','Track Complaint','Service Feedback']
    for p,k in zip(pages,['overview','submit','track','feedback']):
        if st.button(t(k),key='nav_'+p,use_container_width=True,type='primary' if st.session_state.page==p else 'secondary'):go(p)
    st.divider()
    lang=st.selectbox(t('language'),['en','ar'],format_func=lambda x:'English' if x=='en' else 'العربية',index=['en','ar'].index(st.session_state.lang),key='language_selector')
    if lang!=st.session_state.lang:st.session_state.lang=lang;st.rerun()
    st.caption('Sadat Academy for Administrative Sciences' if st.session_state.lang=='en' else 'أكاديمية السادات للعلوم الإدارية')

rtl=st.session_state.lang=='ar'
st.markdown(f'<div class="sadat-direction" dir="{"rtl" if rtl else "ltr"}"></div>',unsafe_allow_html=True)
st.markdown(f'<div class="sadat-topline">SADAT / {html.escape(t({"Overview":"overview","Submit Complaint":"submit","Track Complaint":"track","Service Feedback":"feedback"}[st.session_state.page]))}</div>',unsafe_allow_html=True)
if st.session_state.page=='Overview':
    st.markdown(f'<section class="sadat-hero" dir="{"rtl" if rtl else "ltr"}"><span class="sadat-eyebrow">{t("welcome")}</span><h1>{t("headline")}</h1><p>{t("intro")}</p></section>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a:
        panel('01 · '+t('start'),t('step1'))
        if st.button(t('start'),key='start',type='primary',use_container_width=True):go('Submit Complaint')
    with b:
        panel('02 · '+t('tracknow'),t('step4'))
        if st.button(t('tracknow'),key='tracknow',use_container_width=True):go('Track Complaint')
    with c:
        panel('03 · '+t('rate'),t('feedbackintro'))
        if st.button(t('rate'),key='ratenow',use_container_width=True):go('Service Feedback')
    st.markdown('### '+t('journey'))
    cols=st.columns(4)
    for col,key in zip(cols,['step1','step2','step3','step4']):
        with col:panel(t(key))
    st.info(t('note'))
elif st.session_state.page=='Submit Complaint':
    st.title(t('formtitle'));st.caption(t('formintro'))
    if st.session_state.receipt:
        r=st.session_state.receipt
        st.success(t('success'))
        panel(t('receiptinfo'),t('private'),'success')
        st.code(r['ref'],language=None)
        st.code(r['code'],language=None)
        st.write('**'+t('destination')+':** '+UNITS[r['label']][1 if rtl else 0])
        st.write('**'+t('timestamp')+':** '+r['time'])
        a,b=st.columns(2)
        with a:
            if st.button(t('tracknow'),type='primary',use_container_width=True):go('Track Complaint')
        with b:
            if st.button(t('new'),use_container_width=True):
                st.session_state.receipt=None;st.session_state.draft=None;st.session_state.request_id=None
                for k in ('complaint_name','complaint_email','complaint_body','review_summary'):st.session_state.pop(k,None)
                st.rerun()
    elif st.session_state.draft is None:
        with st.container(border=True):
            with st.form('complaint_form'):
                name=st.text_input(t('name'),key='complaint_name',max_chars=100)
                email=st.text_input(t('email'),key='complaint_email',max_chars=254)
                body=st.text_area(t('body'),key='complaint_body',height=225,max_chars=5000,placeholder=t('placeholder'))
                submitted=st.form_submit_button(t('analyze'),type='primary',use_container_width=True)
            if submitted:
                if not name.strip() or not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+',email.strip()) or len(body.strip())<12:st.error(t('required'))
                else:
                    try:
                        with st.spinner(t('loading')):label,summary,compressed=infer(body.strip())
                        st.session_state.draft=(name.strip(),email.strip(),body.strip(),label,compressed)
                        st.session_state.review_summary=summary
                        st.session_state.request_id=secrets.token_hex(16)
                        st.rerun()
                    except Exception:st.error(t('failure'))
    else:
        name,email,body,label,compressed=st.session_state.draft
        st.subheader(t('review'))
        with st.container(border=True):
            st.markdown('**'+t('original')+'**');st.write(body)
            st.text_area(t('summary'),key='review_summary',height=125,help=t('summaryhelp'),max_chars=5000)
            st.markdown('**'+t('destination')+'**')
            st.info(UNITS[label][1 if rtl else 0]+' · '+t('proposed'))
            a,b=st.columns(2)
            with a:
                if st.button(t('edit'),use_container_width=True):st.session_state.draft=None;st.rerun()
            with b:
                if st.button(t('confirm'),type='primary',use_container_width=True):
                    summary=st.session_state.review_summary.strip()
                    if not summary:st.error(t('required'))
                    else:
                        try:
                            ref,code=save(name,email,body,label,summary,compressed,st.session_state.request_id)
                            if code is None:st.error(t('failure'))
                            else:
                                st.session_state.receipt={'ref':ref,'code':code,'label':label,'time':now()}
                                st.session_state.draft=None;st.rerun()
                        except Exception:st.error(t('failure'))
elif st.session_state.page=='Track Complaint':
    st.title(t('trackingtitle'));st.caption(t('trackingintro'))
    with st.form('tracking_form'):
        ref=st.text_input(t('reference'),key='track_ref')
        code=st.text_input(t('code'),type='password',key='track_code')
        find=st.form_submit_button(t('find'),type='primary')
    if find:
        row,events=lookup(ref,code)
        st.session_state.tracking_result=(row,events) if row else None
        if not row:st.error(t('invalid'))
    if st.session_state.tracking_result:
        row,events=st.session_state.tracking_result
        st.success(t('received'))
        with st.container(border=True):
            st.write('**'+t('reference')+':** '+row['reference'])
            st.write('**'+t('timestamp')+':** '+row['created_at'])
            st.write('**'+t('destination')+':** '+UNITS.get(row['unit'],(row['unit'],row['unit']))[1 if rtl else 0])
            st.caption(t('proposed'))
            st.markdown('**'+t('original')+'**');st.write(row['body'])
            st.markdown('**'+t('summary')+'**');st.write(row['summary'])
            st.subheader(t('timeline'))
            for e in events:
                st.markdown(f'<div class="sadat-event"><b>{html.escape(e["created_at"])}</b><p>{html.escape(t("saved") if e["action"]=="Submitted" else t("thanks"))}</p></div>',unsafe_allow_html=True)
            st.caption(t('noactivity'))
else:
    st.title(t('feedbacktitle'));st.caption(t('feedbackintro'))
    with st.container(border=True):
        with st.form('feedback_form'):
            ref=st.text_input(t('reference'),key='feedback_ref')
            code=st.text_input(t('code'),type='password',key='feedback_code')
            rating=st.select_slider(t('rating'),options=[1,2,3,4,5],value=5)
            comment=st.text_area(t('comment'),max_chars=2000,key='feedback_comment')
            send=st.form_submit_button(t('sendrating'),type='primary')
        if send:
            row,_=lookup(ref,code)
            if not row:st.error(t('invalid'))
            else:
                with connect() as c:
                    cur=c.execute('UPDATE cases SET rating=?,comment=? WHERE id=? AND rating IS NULL',(rating,comment,row['id']))
                    if cur.rowcount:c.execute('INSERT INTO events(case_id,action,note,created_at) VALUES (?,?,?,?)',(row['id'],'Experience feedback','Student rated submission experience.',now()))
                if cur.rowcount:st.success(t('thanks'))
                else:st.warning(t('already'))
