<template>
	<div class='details-table'>
		<b-pagination align="center" size="sm" limit="10" :total-rows="(items != undefined) ? items.length : 0" v-model="current_page" :per-page="7"></b-pagination>
		<b-table striped hover :items="items" :fields="listing_fields" :per-page="7" :current-page="current_page">
			<template slot="show_details" slot-scope="row">
				<b-form-checkbox v-model="row.item._showDetails"></b-form-checkbox>
			</template>
			<template slot="row-details" slot-scope="row">
				<b-card>
					<p v-text="row.item.full"></p>
				</b-card>
			</template>
		</b-table>
		<b-pagination align="center" size="sm" limit="10" :total-rows="(items != undefined) ? items.length : 0" v-model="current_page" :per-page="7"></b-pagination>
	</div>
</template>

<script>
export default {
	name: 'DetailsTable',
	props: ['items'],
	data () {
		return {
			listing_fields: ['content', 'score', 'show_details'],
			current_page: 1,
		}
	}
}
</script>

<style scoped>

</style>
