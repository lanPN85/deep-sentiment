<template>
<div id="sidebar">
    <h5>Upload a file</h5>
    <p>Choose a .txt, .csv or .tsv file. If your file is .txt, each line should be a document. 
        Otherwise, the last column of your file should contain its documents.</p>
    <b-form-file v-model="file"></b-form-file>
    <b-button block variant="success" id="upload-btn" v-if="file" v-on:click="fetchData">Upload</b-button>
    <b-button block variant="success" id="upload-btn" disabled v-else v-on:click="fetchData">Upload & Display</b-button>

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
            source_name: null,
            listing: null,
            listing_fields: ['content', 'score']
        }
    },
    methods: {
        fetchData: function() {
            
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
