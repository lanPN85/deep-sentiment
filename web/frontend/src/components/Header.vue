<template>
<div id="site-header">
    <h2>Deep Sentiment</h2>
    <p><b>Sentiment analysis with deep learning</b></p>
    <p class="subtitle text-right">By Phan Ngoc Lan, Nguyen Duy Manh & Thieu Hai Hoan</p>
</div>
</template>

<script>
export default {
    name: 'SiteHeader',
}
</script>

<style scoped>
.subtitle {
    font-size: smaller;
}

#site-header {
    background-color: #373737;
    padding: 2px 5px 2px 15px;
}

#site-header h2, p {
    color: white;
}

#site-header h2 {
    font-weight: bold;
}
</style>

