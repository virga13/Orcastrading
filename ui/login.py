"""
ui/login.py — Streamlit login / register page.

Rendered when the user is not authenticated. Calls core.auth helpers
and writes session state on success. app.py calls render_login_page()
then st.stop() so no other content is shown while logged out.
"""
from __future__ import annotations
import streamlit as st
from core.auth import login, register, reset_password


def render_login_page() -> None:
    """Render the full-screen login/register UI."""
    st.markdown("""
    <style>
    /* Hide the Streamlit sidebar and header while on login page */
    [data-testid="stSidebar"]{display:none}
    #MainMenu{display:none}
    header[data-testid="stHeader"]{display:none}
    .block-container{max-width:440px!important;padding-top:5rem!important}
    </style>
    """, unsafe_allow_html=True)

    # Logo
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem">
      <div style="font-size:2rem;font-weight:800;color:#F1F5F9;letter-spacing:-.04em">
        Orca<span style="background:linear-gradient(135deg,#3B82F6,#8B5CF6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">trading</span>
      </div>
      <div style="color:#475569;font-size:.85rem;margin-top:.3rem">
        Research platform for serious traders
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Log In", "Create Account"])

    # ── Log In tab ────────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                with st.spinner("Signing in…"):
                    err = login(email, password)
                if err:
                    st.error(err)
                else:
                    st.rerun()

        # Forgot password (outside form to avoid submission conflict)
        with st.expander("Forgot password?"):
            fp_email = st.text_input("Your email address", key="fp_email",
                                     placeholder="you@example.com")
            if st.button("Send reset link", key="fp_btn"):
                if not fp_email:
                    st.warning("Enter your email first.")
                else:
                    err = reset_password(fp_email)
                    if err:
                        st.error(err)
                    else:
                        st.success("Password reset email sent — check your inbox.")

    # ── Create Account tab ────────────────────────────────────────────────────
    with tab_register:
        with st.form("register_form"):
            r_email    = st.text_input("Email", key="r_email",
                                       placeholder="you@example.com")
            r_pass     = st.text_input("Password", type="password", key="r_pass",
                                       placeholder="At least 6 characters")
            r_confirm  = st.text_input("Confirm password", type="password", key="r_confirm",
                                       placeholder="Repeat password")
            r_submitted = st.form_submit_button("Create Account", type="primary",
                                                use_container_width=True)

        if r_submitted:
            if not r_email or not r_pass or not r_confirm:
                st.error("Please fill in all fields.")
            elif r_pass != r_confirm:
                st.error("Passwords do not match.")
            elif len(r_pass) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                with st.spinner("Creating account…"):
                    result = register(r_email, r_pass)
                if result == "CHECK_EMAIL":
                    st.success(
                        "Account created! Check your inbox for a verification link, "
                        "then come back and log in."
                    )
                elif result is None:
                    st.rerun()   # auto-confirmed, session set
                else:
                    st.error(result)

    st.markdown(
        "<div style='text-align:center;margin-top:2rem;color:#334155;font-size:.75rem'>"
        "By signing up you agree to use this tool for research purposes only. "
        "Not financial advice.</div>",
        unsafe_allow_html=True,
    )
