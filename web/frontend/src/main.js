// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import Vuex from 'vuex'
import BootstrapVue from 'bootstrap-vue'
import Icon from 'vue-awesome/components/Icon'

Vue.use(Vuex);
Vue.use(BootstrapVue);
Vue.component('icon', Icon);

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

import App from './App'

Vue.config.productionTip = false

const store = new Vuex.Store({
  state: {
    ready: false
  },
  mutations: {
    getReady(state) {
      state.ready = true
    },
    getUnready(state) {
      state.ready = false
    }
  },
  getters: {
    isReady(state) {
      return state.ready
    }
  }
});

/* eslint-disable no-new */
new Vue({
  el: '#app',
  store,
  template: '<App/>',
  components: { App }
})
