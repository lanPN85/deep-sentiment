<template>
<div id="graphview">
    <h1 class="text-center" v-if="summaryWrapper">Sentiment Distribution</h1>
    <PieChart :options="{responsive: true, maintainAspectRatio: false}" :chart-data="summaryWrapper" id="main-chart" v-if="summaryWrapper"/>
    
    <b-row v-if="isReady">
        <b-col cols="6" offset="3">
            <img src="../assets/ratingbar.png" height="40vw" width="100%"/>
        </b-col>
    </b-row>
    <b-row v-if="isReady">
        <b-col cols="6" offset="3">
            <p class="text-center rb-subtitle">Rating by sentiment score</p>
        </b-col>
    </b-row>
</div>
</template>

<script>
import $ from 'jquery'
import 'vue-awesome/icons/refresh'
import PieChart from './PieChart'
import { mapGetters } from 'vuex'

export default {
    name: 'GraphView',
    components: {
        PieChart
    },
    data() {
        return {
            summary: null,
            summaryWrapper: null
        }
    },
    computed: {
        ...mapGetters(['isReady'])
    },
    watch: {
        isReady: function(val) {
            if (val) this.fetchData();
            else {
                this.summaryWrapper = null;
                this.summary = null;
            }    
        }
    },
    methods: {
        fillData: function() {
            if (!this.summary) {
                return;
            }

            this.summaryWrapper = {
                labels: ['Positive', 'Negative', 'Unsure/None'],
                datasets: [{
                        data: [this.summary.positive, this.summary.negative, this.summary.unsure],
                        backgroundColor: ['#81c784', '#e57373', '#9e9e9e'],
                        hoverBackgroundColor: ['#519657', '#af4448', '#707070']
                }]
            }
        },
        fetchData: function() {
            var component = this;
            $.getJSON('/api/summary', function(data) {
                component.summary = data;
                component.fillData();
            });
        }
    }
}
</script>

<style scoped>
#graphview {
    padding: 15px 0 5px 0;
}

#main-chart {
    margin-top: 10px;
    margin-bottom: 20px;
    min-height: 65vh;
}

#bar-wrap {
    height: 5px;
}

.rb-subtitle {
    font-size: 11pt;
}
</style>
