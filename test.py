import streamlit as st

st.title("🧾 Invoice / Receipt Field Extractor")
col1, col2 = st.columns(2)
with col1:
    st.header("masala tea")
    vote1=st.button("Vote for masala tea")
with col2:
    st.header("adrak chai")
    vote2=st.button("Vote for adrak chai")

if vote1:
    st.success("You voted for masala tea!")
elif vote2:
    st.success("You voted for adrak chai!")