<template>
<div id="sidebar">
    <h5>Upload a file</h5>
    <p>Choose a .txt, .csv or .tsv file. If your file is .txt, each line should be a document. 
        Otherwise, the last column of your file should contain its documents.</p>
    <b-form-file v-model="file"></b-form-file>
    <b-button block variant="success" id="upload-btn" v-if="file">Upload</b-button>
    <b-button block variant="success" id="upload-btn" disabled v-else>Upload & Display</b-button>

    <div id="detail-pane" v-if="listing">
        <h5>Details</h5>
        <p>File: {{source_name}}</p>
        <b-tabs>
            <b-tab title="Positive">
                <b-table striped hover :items="listing.positive" :fields="listing_fields" no-provider-paging="true"></b-table>
            </b-tab>
            <b-tab title="Negative">
                <b-table striped hover :items="listing.negative" :fields="listing_fields" no-provider-paging="true"></b-table>
            </b-tab>
            <b-tab title="Unsure/None">
                <b-table striped hover :items="listing.unsure" :fields="listing_fields" no-provider-paging="true"></b-table>
            </b-tab>
        </b-tabs>
    </div>
</div>
</template>

<script>
export default {
    name: 'Sidebar',
    data() {
        return {
            file: null,
            source_name: 'test.txt',
            listing: {'positive': [{'content': 'the most "spiritual" film I have seen in a long long time. maybe ever. also one scene around the dining room table a piece of comic perfection. I understand a release date is coming up in the fall. if it comes to your town and you want to see a movie that makes you think this is it. Aviva is great in it and she is most certainly a future star - (Superbad is out now which she is in) - also all the actors seem perfectly calibrated. There is a tone set by this movie that is used to surprise through out. i would not know what to call it - it is comedy but the undergirding message is so fierce and direct that "comedy" is not a big enough word for it. I love this film. It is a thinking man\'s comedy. but even that phrase is not really good enough.', 'score': 0.9}],
                'negative': [{'content': 'Hello', 'score': 0.2}],
                'unsure': [{'content': 'Hello', 'score': 0.5}]},
            listing_fields: ['content', 'score']
        }
    }
}
</script>

<style scoped>
#sidebar {
    margin-top: 5px;
    padding: 0 5px 3px 5px;
    border-style: solid;
    border-color: black;
    border-width: 0 1px 0 0;
    height: 90vh;
    min-height: 90vh;
}

#upload-btn {
    margin-top: 5px;
}

#detail-pane {
    margin-top: 25px;
}

h5 {
    font-weight: bold;
}
</style>
