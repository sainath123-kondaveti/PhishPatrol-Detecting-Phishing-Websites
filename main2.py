import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
import re
import socket

# Load the trained model
model = joblib.load('voting_classifier.pkl')

# Extract features from the given URL
def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    try:
        ip = socket.gethostbyname(hostname)
        is_ip = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname) else 0
    except:
        ip = ""
        is_ip = 0

    features = {
        'NumDots': url.count('.'),
        'SubdomainLevel': hostname.count('.') - 1,
        'PathLevel': path.count('/'),
        'UrlLength': len(url),
        'NumDash': url.count('-'),
        'NumDashInHostname': hostname.count('-'),
        'AtSymbol': 1 if '@' in url else 0,
        'TildeSymbol': 1 if '~' in url else 0,
        'NumUnderscore': url.count('_'),
        'NumPercent': url.count('%'),
        'NumQueryComponents': query.count('&'),
        'NumAmpersand': url.count('&'),
        'NumHash': url.count('#'),
        'NumNumericChars': sum(c.isdigit() for c in url),
        'NoHttps': 1 if parsed.scheme != 'https' else 0,
        'RandomString': 1 if re.search(r'[a-z]{10,}', hostname) else 0,
        'IpAddress': is_ip,
        'DomainInSubdomains': 1 if any(part in hostname.split('.') for part in ['com', 'net', 'org']) else 0,
        'DomainInPaths': 1 if any(part in path for part in ['com', 'net', 'org']) else 0,
        'HttpsInHostname': 1 if 'https' in hostname else 0,
        'HostnameLength': len(hostname),
        'PathLength': len(path),
        'QueryLength': len(query),
        'DoubleSlashInPath': 1 if '//' in path else 0,
        'NumSensitiveWords': 1 if any(word in url.lower() for word in ['secure', 'account', 'banking', 'confirm', 'verify']) else 0,
        
        # Neutral/default values
        'EmbeddedBrandName': 0,
        'PctExtHyperlinks': 0.0,
        'PctExtResourceUrls': 0.0,
        'ExtFavicon': 0,
        'InsecureForms': 0,
        'RelativeFormAction': 0,
        'ExtFormAction': 0,
        'AbnormalFormAction': 0,
        'PctNullSelfRedirectHyperlinks': 0.0,
        'FrequentDomainNameMismatch': 0,
        'FakeLinkInStatusBar': 0,
        'RightClickDisabled': 0,
        'PopUpWindow': 0,
        'SubmitInfoToEmail': 0,
        'IframeOrFrame': 0,
        'MissingTitle': 0,
        'ImagesOnlyInForm': 0,
        'SubdomainLevelRT': 0,
        'UrlLengthRT': 0,
        'PctExtResourceUrlsRT': 0.0,
        'AbnormalExtFormActionR': 0,
        'ExtMetaScriptLinkRT': 0,
        'PctExtNullSelfRedirectHyperlinksRT': 0.0
    }

    return features

# Streamlit UI
st.set_page_config(page_title="PhishPatrol", layout="centered")
st.title("🔒 PhishPatrol")
st.write("A Machine Learning Powered Tool to Detect Suspicious URLs")

url = st.text_input("🔍 Enter URL to analyze:", "https://www.google.com")

if st.button("🔎 Analyze"):
    try:
        features = extract_features(url)
        input_df = pd.DataFrame([features])
        
        # Count suspicious features
        suspicious_features = [k for k, v in features.items() if v == 1]
        total_features = len(features)
        suspicious_ratio = len(suspicious_features) / total_features

        # Predict using the model
        model_prediction = model.predict(input_df)[0]

        # Apply 10% rule
        if suspicious_ratio > 0.10 or model_prediction == 1:
            st.error("⚠️ Phishing Website Detected")
            st.markdown("#### ⚠️ Suspicious Features Detected:")
            st.write(suspicious_features)
        else:
            st.success("✅ Legitimate Website")

        with st.expander("🧪 Feature Details"):
            for feature, value in features.items():
                if value == 1:
                    st.markdown(f"- **{feature}**: ⚠️ Suspicious")
                elif value == 0:
                    st.markdown(f"- **{feature}**: ✅ Safe")
                else:
                    st.markdown(f"- **{feature}**: ℹ️ Neutral ({value})")

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.sidebar.info("👨‍💻 This tool uses a voting classifier trained on phishing datasets and flags URLs as phishing if more than 10% of features are suspicious.")
