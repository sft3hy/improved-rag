import streamlit as st
from streamlit_js_eval import streamlit_js_eval


def get_user_info():
    """
    Gathers a comprehensive set of user information from their browser.

    This function collects the user's IP address (if available through Streamlit's
    context), and a wealth of browser-specific details by executing Javascript.

    Returns:
        dict: A dictionary containing various pieces of user information.
              Returns an empty dictionary if an error occurs during information
              retrieval.
    """
    user_info = {}
    try:
        # Get IP address from Streamlit's context if available
        # Note: This might not always be the true client IP due to proxies.
        if hasattr(st, "context") and hasattr(st.context, "ip_address"):
            user_info["ip"] = st.context.ip_address

        # Use streamlit-js-eval to get detailed browser information
        js_code = """
        () => {
            return {
                userAgent: navigator.userAgent,
                language: navigator.language,
                platform: navigator.platform,
                screenWidth: screen.width,
                screenHeight: screen.height,
                colorDepth: screen.colorDepth,
                pixelRatio: window.devicePixelRatio,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                cookiesEnabled: navigator.cookieEnabled,
                doNotTrack: navigator.doNotTrack,
                hardwareConcurrency: navigator.hardwareConcurrency,
                vendor: navigator.vendor,
                plugins: Array.from(navigator.plugins).map(p => p.name)
            };
        }
        """
        browser_info = streamlit_js_eval(js_expressions=js_code, want_output=True)
        print(browser_info)
        if browser_info:
            user_info.update(browser_info)

    except Exception as e:
        st.error(f"An error occurred while fetching user information: {e}")
        return {}

    return user_info


def float_to_percent(score) -> str:
    # Ensure score is a float
    if type(score) == list:
        score = score[0]
        return score
    score = float(score)
    # Convert to percent and round
    percent = round(score * 100, 1)
    return f"{percent}%"
