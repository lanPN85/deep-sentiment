<template>
<div id="sidebar">
    <h5>Upload a file</h5>
    <p>Choose a .txt, .csv or .tsv file. If your file is .txt, each line should be a document. 
        Otherwise, the last column of your file should contain its documents.</p>
    <b-form-file v-model="file"></b-form-file>
    <b-button block variant="success" id="upload-btn" :disabled="!file || isLoading" v-on:click="fetchData">Upload & Display</b-button>
    <p v-if="isLoading" id="loading-prompt">
        <icon name="circle-o-notch" scale="0.75" spin></icon> Processing file. This may take several minutes.
    </p>

    <div id="detail-pane" v-if="isReady">
        <h5>Details</h5>
        <b-tabs>
            <b-tab title="Positive">
                <details-table :items="listing.positive"></details-table>
            </b-tab>
            <b-tab title="Negative">
                <details-table :items="listing.negative"></details-table>
            </b-tab>
            <b-tab title="Unsure/None">
                <details-table :items="listing.unsure"></details-table>
            </b-tab>
        </b-tabs>
    </div>
</div>
</template>

<script>
import 'vue-awesome/icons/circle-o-notch'
import $ from 'jquery'
import DetailsTable from './DetailsTable'
import { mapGetters } from 'vuex'

export default {
    name: 'Sidebar',
    data() {
        return {
            file: null,
            source_name: null,
            listing: null,
            isLoading: false
        }
    },
    computed: {
        ...mapGetters(['isReady'])
    },
    components: {
        DetailsTable
    },
    methods: {
        fetchData: function() {
            var component = this;
            var formData = new FormData();
            this.source_name = this.file.name;
            this.listing = null;
            this.isLoading = true;
            this.$store.commit('getUnready');

            formData.append('file', this.file, this.file.name);

            $.ajax({
                url: '/api/upload/',
                data: formData,
                timeout: 60000,
                type: 'POST',
                success: function(data) {
                    if (data.error) {
                        alert(data.error);
                    } else {
                        $.getJSON('/api/listing/positive', function(data){
                            var tmp_listing = {};
                            tmp_listing.positive = data;
                            $.getJSON('/api/listing/negative', function(data){
                                tmp_listing.negative = data;
                                $.getJSON('/api/listing/unsure', function(data){
                                    tmp_listing.unsure = data;
                                    component.listing = tmp_listing;
                                    component.isLoading = false;
                                    component.$store.commit('getReady');
                                });
                            });
                        });
                    }
                }, 
                processData: false,
                contentType: false
            });
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
    max-height: 95vh;
    overflow: scroll;
}

#upload-btn {
    margin-top: 5px;
}

#detail-pane {
    margin-top: 25px;
}

#loading-prompt {
    margin-top: 5px;
    color: #c62828;
}

h5 {
    font-weight: bold;
}
</style>
